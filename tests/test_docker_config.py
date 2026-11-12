"""Static checks of the container configuration: Dockerfile, .dockerignore and deployment/gunicorn.conf.py.

Docker is not required: the .dockerignore rules are evaluated with an emulation of Docker's matcher (moby
patternmatcher: patterns in order, the last match wins, a pattern also matches a path's parent directories,
`*` stays within one path segment and `**` spans segments). These tests do not prove that the image builds or
runs on Linux -- only that the configuration says what it is meant to say.
"""
import importlib.util
import json
import os
import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
GUNICORN_CONF = REPO_ROOT / "deployment" / "gunicorn.conf.py"
_spec = importlib.util.spec_from_file_location("prepare_deployment_data", REPO_ROOT / "scripts" / "prepare_deployment_data.py")
pdd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdd)

TRACKED_CONFIG = {
    "config/data_config.yaml", "config/cf_config.yaml", "config/kg_config.yaml", "config/hybrid_config.yaml",
    "results/day4/selected_cbf_config.json", "results/day5/selected_cf_config.json",
    "results/day7/selected_kg_config.json", "results/day7/graph_snapshot.json",
    "results/day8/selected_hybrid_config.json", "results/day8/day8_manifest.json",
}
PACKAGE = "deployment/runtime-data"
SKIP_WALK = {".git", ".venv", "venv", "env", "raw", "synthetic_private", "node_modules"}  # never listed, never needed


# ---------------------------------------------------------------------------
# .dockerignore emulation
# ---------------------------------------------------------------------------
def _pattern_regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def dockerignore_rules():
    rules = []
    for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        cleaned = str(PurePosixPath(line.lstrip("!").strip("/")))
        rules.append((negate, _pattern_regex(cleaned)))
    return rules


def excluded(path: str, rules) -> bool:
    candidates = [path] + [str(p) for p in PurePosixPath(path).parents if str(p) != "."]
    result = False
    for negate, regex in rules:
        if any(regex.match(c) for c in candidates):
            result = not negate
    return result


def build_context(rules) -> set[str]:
    included = set()
    for current, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_WALK]
        for name in files:
            rel = (Path(current) / name).relative_to(REPO_ROOT).as_posix()
            if not excluded(rel, rules):
                included.add(rel)
    return included


def tracked(prefix: str) -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", prefix], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return {p for p in out.stdout.splitlines() if "__pycache__" not in p}


class DockerignoreTests(unittest.TestCase):
    rules = dockerignore_rules()

    def test_emulation_follows_docker_semantics(self):
        r = [(False, _pattern_regex("*")), (True, _pattern_regex("src")), (False, _pattern_regex("**/__pycache__"))]
        self.assertFalse(excluded("src/app/factory.py", r))
        self.assertTrue(excluded("src/app/__pycache__/factory.cpython-313.pyc", r))
        self.assertTrue(excluded("docs/deployment.md", r))
        self.assertTrue(excluded("a/b/c.txt", [(False, _pattern_regex("a"))]))  # parents match

    def test_sensitive_and_unneeded_paths_are_never_in_the_context(self):
        for path in (".env", ".env.example", "src/.env", "deployment/runtime-data/.env", ".venv/Lib/site.py", ".git/config",
                     "data/raw/upwork-jobs.csv", "data/processed/synthetic_private/truth.csv",
                     "deployment/runtime-data/data/processed/synthetic_private/truth.csv",
                     "data/processed/sampled_requests.csv", "tests/test_app_unit.py", "docs/deployment.md",
                     "scripts/load_postgres.py", "notebooks/x.ipynb", "results/day8/day8_metrics.csv",
                     "config/generator_v1.yaml", "requirements.txt", "requirements-dev.txt", "backup.dump", "local.db",
                     "deployment/runtime-data/data/processed/dump.sql", ".pytest_cache/v/cache", "Dockerfile"):
            self.assertTrue(excluded(path, self.rules), path)

    def test_context_is_exactly_the_application_and_its_runtime_files(self):
        context = build_context(self.rules)
        expected = (tracked("src") | TRACKED_CONFIG | {"requirements-prod.lock", "deployment/gunicorn.conf.py"})
        package = {f"{PACKAGE}/{rel}" for rel in pdd.REQUIRED_FILES} | {f"{PACKAGE}/{pdd.PACKAGE_MANIFEST}"}
        if (REPO_ROOT / PACKAGE).is_dir():
            expected |= package
        self.assertEqual(context - expected, set(), "unexpected files would be sent to the build")
        self.assertEqual(expected - context, set(), "required files would be missing from the build")


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------
def instructions():
    joined = re.sub(r"\\\n", " ", DOCKERFILE)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


class DockerfileTests(unittest.TestCase):
    def test_base_image_is_pinned_debian_slim_python_313(self):
        stages = [i for i in instructions() if i.startswith("FROM ")]
        self.assertEqual(len(stages), 2)  # app-files (discarded) and the runtime image
        image = r"python:3\.13\.\d+-slim-bookworm@sha256:[0-9a-f]{64}"
        self.assertRegex(stages[0], rf"^FROM {image} AS app-files$")
        self.assertRegex(stages[1], rf"^FROM {image}$")
        self.assertEqual(stages[0].split()[1], stages[1].split()[1])  # both stages: the same pinned image
        self.assertNotIn("alpine", DOCKERFILE.lower())

    def test_only_locked_production_dependencies_are_installed(self):
        pip = [i for i in instructions() if "pip install" in i]
        self.assertEqual(pip, ["RUN pip install --no-deps --require-hashes --only-binary=:all: -r requirements-prod.lock"])
        self.assertNotIn("requirements-dev", DOCKERFILE)
        self.assertNotIn("requirements.txt", DOCKERFILE)

    def test_time_zone_database_is_installed(self):
        self.assertRegex(DOCKERFILE, r"apt-get install -y --no-install-recommends tzdata")

    def test_runs_as_non_root_with_the_gunicorn_config(self):
        steps = instructions()
        users = [i for i in steps if i.startswith("USER ")]
        self.assertEqual(users, ["USER hirewise"])
        self.assertLess(steps.index("USER hirewise"), next(i for i, s in enumerate(steps) if s.startswith("CMD ")))
        self.assertEqual(steps[-1], 'CMD ["gunicorn", "--config", "deployment/gunicorn.conf.py", "src.app:create_app()"]')

    def test_every_copy_source_is_in_the_build_context(self):
        rules = dockerignore_rules()
        for step in instructions():
            if step.startswith("COPY ") and not step.startswith("COPY --from="):
                *sources, _ = step.split()[1:]
                for src in sources:
                    self.assertFalse(excluded(src.rstrip("/"), rules), f"COPY source excluded by .dockerignore: {src}")
                    self.assertFalse(src.startswith(("data/", ".env", "tests", "scripts")), src)
        self.assertFalse([i for i in instructions() if i.startswith("ADD ")])

    def test_no_secrets_ports_or_urls_are_built_in(self):
        env_lines = " ".join(i for i in instructions() if i.startswith(("ENV ", "ARG ")))
        for word in ("SECRET", "PASSWORD", "TOKEN", "CLIENT_ID", "POSTGRES", "SMTP", "NEO4J", "APP_BASE_URL", "PORT="):
            self.assertNotIn(word, env_lines.upper())
        self.assertFalse([i for i in instructions() if i.startswith("EXPOSE ")])

    def test_application_files_are_readable_but_not_writable_by_the_app_user(self):
        steps = instructions()
        runtime_start = [i for i, s in enumerate(steps) if s.startswith("FROM ")][1]
        stage1, stage2 = steps[:runtime_start], steps[runtime_start:]
        chmod = "RUN chmod -R u=rwX,go=rX /app"
        self.assertIn(chmod, stage1)
        last_copy = max(i for i, s in enumerate(stage1) if s.startswith("COPY "))
        self.assertGreater(stage1.index(chmod), last_copy)  # applied after every file is in place
        copies = [s for s in stage2 if s.startswith("COPY ")]
        self.assertEqual(copies, ["COPY requirements-prod.lock ./", "COPY --from=app-files /app/ /app/"])
        self.assertNotRegex(DOCKERFILE, r"--chown|chown |chmod [^-]*[0-7]*7[0-7]*\b|a\+w|o\+w|go\+w")
        # chmod's symbolic rule, applied to modes a build host might produce: the app user ("other") can read every
        # file and traverse every directory, and can write nothing.
        def normalise(mode, is_dir):
            x = 0o111 if (is_dir or mode & 0o111) else 0
            return 0o600 | (0o100 if x else 0) | 0o044 | (0o011 if x else 0)
        for mode in (0o600, 0o640, 0o644, 0o700, 0o755, 0o777, 0o400):
            for is_dir in (False, True):
                result = normalise(mode, is_dir)
                self.assertTrue(result & 0o004, oct(mode))                       # readable by others
                self.assertFalse(result & 0o022, oct(mode))                      # writable by root only
                if is_dir:
                    self.assertTrue(result & 0o001, oct(mode))                   # traversable by others

    def test_health_check_uses_the_existing_endpoint(self):
        [check] = [i for i in instructions() if i.startswith("HEALTHCHECK ")]
        self.assertIn("/api/health", check)
        self.assertIn("os.environ['PORT']", check)

    def test_inline_python_snippets_compile_and_the_data_check_passes_on_the_package(self):
        snippets = re.findall(r'python -c "(.*?)"', re.sub(r"\\\n", " ", DOCKERFILE), flags=re.S)
        self.assertEqual(len(snippets), 2)  # the runtime-data check and the health check
        for code in snippets:
            compile(code, "<Dockerfile>", "exec")
        check = next(s for s in snippets if "runtime data" in s)
        package = REPO_ROOT / PACKAGE
        if not (package / pdd.PACKAGE_MANIFEST).is_file():
            self.skipTest("runtime-data package not present")
        local = check.replace("/tmp/runtime-data-manifest.json", str((package / pdd.PACKAGE_MANIFEST).as_posix()))
        cwd = os.getcwd()
        try:  # read-only: hashes the packaged files in place, exactly as the build step does in /app
            os.chdir(package)
            with mock.patch("builtins.print"):
                exec(compile(local.replace("str(p)", "p.as_posix()"), "<Dockerfile>", "exec"), {})
        finally:
            os.chdir(cwd)

    def test_build_rechecks_the_runtime_data_against_its_manifest(self):
        self.assertIn("COPY deployment/runtime-data/data/ data/", DOCKERFILE)
        self.assertIn("runtime data does not match its manifest", DOCKERFILE)
        manifest = REPO_ROOT / PACKAGE / pdd.PACKAGE_MANIFEST
        if manifest.is_file():
            entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
            self.assertEqual(sorted(e["path"] for e in entries), sorted(pdd.REQUIRED_FILES))


# ---------------------------------------------------------------------------
# Gunicorn configuration (plain Python; evaluated without importing gunicorn, which needs POSIX)
# ---------------------------------------------------------------------------
def load_gunicorn_conf(env):
    namespace = {}
    with mock.patch.dict(os.environ, env, clear=True):
        exec(compile(GUNICORN_CONF.read_text(encoding="utf-8"), str(GUNICORN_CONF), "exec"), namespace)
    return namespace


class GunicornConfigTests(unittest.TestCase):
    def test_settings(self):
        conf = load_gunicorn_conf({"PORT": "10000"})
        self.assertEqual(conf["bind"], ["0.0.0.0:10000"])
        self.assertEqual((conf["workers"], conf["worker_class"], conf["threads"], conf["timeout"]), (1, "gthread", 4, 60))
        self.assertIsNone(conf["accesslog"])  # request URLs can carry single-use tokens / OAuth codes
        self.assertFalse(conf["preload_app"])  # the app and its database engine are created after the fork
        self.assertEqual(conf["errorlog"], "-")

    def test_port_is_required_and_validated(self):
        for env in ({}, {"PORT": ""}, {"PORT": "abc"}, {"PORT": "0"}, {"PORT": "70000"}):
            with self.assertRaises(SystemExit, msg=env):
                load_gunicorn_conf(env)


if __name__ == "__main__":
    unittest.main()
