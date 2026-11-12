"""Generate requirements-prod.lock: every production dependency (direct and transitive) pinned, with hashes, for
the Linux x86_64 / CPython 3.13 (glibc) container image.

Usage (from repository root, with the project's tested development environment active):

    python scripts/lock_production_requirements.py           # write requirements-prod.lock
    python scripts/lock_production_requirements.py --check   # verify the existing lock is current; write nothing

Method (no Linux machine is needed, and nothing is installed):
  1. Inputs: the approved direct pins in requirements-prod.txt; every transitive dependency is constrained to the
     version installed (and tested) in the current environment, so nothing is silently upgraded.
  2. pip resolves against Linux wheels only (`pip install --dry-run --report --only-binary=:all: --platform
     manylinux... --python-version 3.13 --abi cp313`).
  3. pip evaluates environment markers for the machine it runs on (e.g. sys_platform == "win32"), so the
     dependency set is recomputed here from each package's Requires-Dist with LINUX markers, every version is
     checked against every requirement's specifier, and missing Linux-only dependencies are fed back to pip.
  4. Hashes: the SHA-256 of every wheel of the locked version that is installable on manylinux x86_64 / CPython
     3.13 (from PyPI's JSON API), so `pip install --require-hashes` accepts whichever compatible wheel it picks.

Install in the image with:  pip install --no-deps --require-hashes -r requirements-prod.lock
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "requirements-prod.txt"
OUTPUT = REPO_ROOT / "requirements-prod.lock"
PYTHON = (3, 13)
GLIBC_MINORS = range(17, 37)  # manylinux_2_17 .. manylinux_2_36 (Debian bookworm, the python:3.13-slim base, has 2.36)
PLATFORMS = [f"manylinux_2_{m}_x86_64" for m in reversed(GLIBC_MINORS)] + ["manylinux2014_x86_64"]
LINUX_ENV = {
    "implementation_name": "cpython", "implementation_version": "3.13.0", "os_name": "posix",
    "platform_machine": "x86_64", "platform_python_implementation": "CPython", "platform_release": "",
    "platform_system": "Linux", "platform_version": "", "python_full_version": "3.13.0", "python_version": "3.13",
    "sys_platform": "linux",
}
TARGET_TAGS = set(cpython_tags(PYTHON, abis=["cp313", "abi3", "none"], platforms=PLATFORMS)) | set(
    compatible_tags(PYTHON, "cp313", PLATFORMS))


class LockError(RuntimeError):
    pass


def direct_requirements() -> list[Requirement]:
    reqs = []
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            req = Requirement(line)
            if len(req.specifier) != 1 or next(iter(req.specifier)).operator != "==":
                raise LockError(f"{INPUT.name}: direct requirement must be pinned with ==: {line}")
            reqs.append(req)
    return reqs


def tested_versions() -> dict[str, str]:
    return {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}


def pip_resolve(direct: list[Requirement], extra: set[str], constraints: dict[str, str]) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        cfile, report = Path(tmp) / "constraints.txt", Path(tmp) / "report.json"
        cfile.write_text("".join(f"{n}=={v}\n" for n, v in sorted(constraints.items())), encoding="utf-8")
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
               "--only-binary=:all:", "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313",
               *[a for p in PLATFORMS for a in ("--platform", p)], "-c", str(cfile), "--report", str(report),
               *[str(r) for r in direct], *sorted(extra)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise LockError("pip could not resolve Linux wheels:\n" + done.stderr.strip())
        data = json.loads(report.read_text(encoding="utf-8"))
    return {canonicalize_name(p["metadata"]["name"]): {"name": p["metadata"]["name"], "version": p["metadata"]["version"],
                                                       "requires": p["metadata"].get("requires_dist") or []}
            for p in data["install"]}


def linux_closure(direct: list[Requirement], resolved: dict[str, dict]) -> tuple[dict[str, str], set[str]]:
    """{package: version} needed on Linux, and the names missing from pip's (marker-skewed) resolution."""
    needed: dict[str, str] = {}
    missing: set[str] = set()
    stack = [(req, f"{INPUT.name}") for req in direct]
    seen = set()
    while stack:
        req, via = stack.pop()
        name = canonicalize_name(req.name)
        if name not in resolved:
            missing.add(name)
            continue
        version = resolved[name]["version"]
        if not req.specifier.contains(version, prereleases=True):
            raise LockError(f"{resolved[name]['name']} {version} does not satisfy '{req}' (required by {via})")
        needed[name] = version
        for extra in sorted(req.extras) + [""]:
            if (name, extra) in seen:
                continue
            seen.add((name, extra))
            for spec in resolved[name]["requires"]:
                dep = Requirement(spec)
                if dep.marker is None or dep.marker.evaluate({**LINUX_ENV, "extra": extra}):
                    stack.append((dep, f"{resolved[name]['name']}[{extra}]" if extra else resolved[name]["name"]))
    return needed, missing


def linux_wheel_hashes(name: str, version: str) -> list[str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as resp:
        files = json.load(resp)["urls"]
    hashes = []
    for f in files:
        if f["packagetype"] != "bdist_wheel" or f.get("yanked"):
            continue
        _, _, _, tags = parse_wheel_filename(f["filename"])
        if tags & TARGET_TAGS:
            hashes.append(f["digests"]["sha256"])
    if not hashes:
        raise LockError(f"no Linux x86_64 / CPython 3.13 wheel on PyPI for {name}=={version}")
    return sorted(hashes)


def build_lock() -> tuple[str, dict]:
    direct = direct_requirements()
    tested = tested_versions()
    direct_names = {canonicalize_name(r.name) for r in direct}
    constraints = {n: v for n, v in tested.items() if n not in direct_names}
    extra: set[str] = set()
    for _ in range(5):
        resolved = pip_resolve(direct, extra, constraints)
        needed, missing = linux_closure(direct, resolved)
        if not missing:
            break
        extra |= missing
    else:
        raise LockError(f"could not close the Linux dependency set; still missing: {sorted(missing)}")
    for req in direct:  # the approved direct pins are never changed
        if needed[canonicalize_name(req.name)] != next(iter(req.specifier)).version:
            raise LockError(f"direct pin changed for {req.name}")
    summary = {
        "locked": {resolved[n]["name"]: v for n, v in sorted(needed.items())},
        "differs_from_tested": {resolved[n]["name"]: (tested.get(n), v) for n, v in needed.items()
                                if tested.get(n) not in (None, v)},
        "not_installed_locally": sorted(resolved[n]["name"] for n in needed if n not in tested),
        "excluded_by_linux_markers": sorted(p["name"] for n, p in resolved.items() if n not in needed),
    }
    lines = [
        "# Fully pinned production dependencies for the Linux x86_64 / CPython 3.13 (glibc) container image.",
        "# Generated by scripts/lock_production_requirements.py from requirements-prod.txt -- do not edit by hand.",
        "# Install with: pip install --no-deps --require-hashes -r requirements-prod.lock",
        "",
    ]
    for n in sorted(needed):
        hashes = linux_wheel_hashes(resolved[n]["name"], needed[n])
        lines.append(f"{resolved[n]['name']}=={needed[n]} \\")
        lines.extend(f"    --hash=sha256:{h}" + (" \\" if i < len(hashes) - 1 else "") for i, h in enumerate(hashes))
    return "\n".join(lines) + "\n", summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify requirements-prod.lock is current; write nothing")
    args = parser.parse_args(argv)
    try:
        text, summary = build_lock()
    except LockError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1))
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        print("lock is current" if current == text else "lock is OUT OF DATE")
        return 0 if current == text else 1
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(summary['locked'])} packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
