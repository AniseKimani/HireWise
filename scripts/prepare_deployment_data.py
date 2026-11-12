"""Package the git-ignored runtime data the deployed recommender needs (deployment preparation).

The frozen recommender reads 14 git-ignored files from data/processed/ once, at start-up (docs/deployment.md).
This script copies exactly those files -- an explicit allow-list, never a recursive copy of data/processed --
into deployment/runtime-data/, keeping their repository-relative paths, so a Docker build can add them
without the rest of the (private or raw) data.

Usage (from repository root):

    python scripts/prepare_deployment_data.py            # verify the sources, then create the package
    python scripts/prepare_deployment_data.py --force    # replace an existing package
    python scripts/prepare_deployment_data.py --verify   # check an existing package; copies nothing

Checks (any failure: exit status 1, a reason, and no package is left behind):
  * every allow-listed source exists, is a regular file inside the repository, and no path component is a
    symlink or junction;
  * every file's SHA-256 matches the checksum recorded by the pipeline that produced it
    (data/processed/manifest.json, data/processed/synthetic/manifest.json, data/processed/kg/kg_prepare_manifest.json);
  * the KG CSVs reproduce the frozen Day 7 graph fingerprint (results/day7/graph_snapshot.json), computed with
    the project's own src.kg.graph_view.load_view_from_kg_csvs -- the same check the application makes at start-up;
  * the package holds exactly the allow-listed files plus its own checksum manifest: nothing else, no symlinks,
    nothing from data/raw or data/processed/synthetic_private.

Nothing under data/ or results/ is ever modified. The 10 tracked configuration/results files the recommender
also reads come from the repository itself and are not part of this package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "data/processed/synthetic/workers.csv",
    "data/processed/synthetic/requests.csv",
    "data/processed/synthetic/interactions.csv",
    "data/processed/sampled_requests.csv",
    "data/processed/prepared_jobs.csv",
    "data/processed/kg/nodes_category.csv",
    "data/processed/kg/nodes_skill.csv",
    "data/processed/kg/nodes_worker.csv",
    "data/processed/kg/rel_has_skill.csv",
    "data/processed/kg/rel_hired.csv",
    "data/processed/kg/rel_interested_in.csv",
    "data/processed/kg/rel_related_to.csv",
    "data/processed/kg/rel_reviewed.csv",
    "data/processed/kg/rel_specializes_in.csv",
)
# Pipeline manifests recording each file's SHA-256 (only the entries for allow-listed files are used).
SOURCE_MANIFESTS = (
    "data/processed/manifest.json",
    "data/processed/synthetic/manifest.json",
    "data/processed/kg/kg_prepare_manifest.json",
)
GRAPH_SNAPSHOT = "results/day7/graph_snapshot.json"
KG_DIR = "data/processed/kg"
PACKAGE_DIR = "deployment/runtime-data"
PACKAGE_MANIFEST = "runtime-data-manifest.json"
FORBIDDEN_PREFIXES = ("data/raw/",)
FORBIDDEN_PARTS = ("synthetic_private",)


class PackagingError(RuntimeError):
    """The sources or the package failed a check; the message says why."""


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def check_allow_list(files=REQUIRED_FILES) -> None:
    """The allow-list itself: relative, normalised, unique, under data/processed, never private or raw."""
    if len(set(files)) != len(files):
        raise PackagingError("the allow-list contains duplicate paths")
    for rel in files:
        posix = PurePosixPath(rel)
        if posix.is_absolute() or ".." in posix.parts or "\\" in rel or str(posix) != rel:
            raise PackagingError(f"allow-list path is not a plain repository-relative path: {rel}")
        if not rel.startswith("data/processed/"):
            raise PackagingError(f"allow-list path is outside data/processed: {rel}")
        if rel.startswith(FORBIDDEN_PREFIXES) or any(part in posix.parts for part in FORBIDDEN_PARTS):
            raise PackagingError(f"allow-list path is private or raw data: {rel}")


def safe_path(root: Path, rel: str, *, must_exist: bool = True) -> Path:
    """root/rel, refusing symlinks/junctions on any component and anything that resolves outside root."""
    root = Path(root)
    path = root
    for part in PurePosixPath(rel).parts:
        path = path / part
        if _is_link(path):
            raise PackagingError(f"refusing symlink or junction: {rel} ({path.relative_to(root).as_posix()})")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise PackagingError(f"path escapes {root}: {rel}") from None
    if must_exist and not path.is_file():
        raise PackagingError(f"required file is missing: {rel}")
    return path


def _package_dir(repo_root: Path, dest: Path) -> Path:
    """The destination must lie inside <repo>/deployment and must not itself be (inside) a symlink."""
    dest = Path(dest)
    deployment = Path(repo_root) / "deployment"
    try:
        rel = dest.resolve().relative_to(deployment.resolve())
    except ValueError:
        raise PackagingError(f"the package directory must be inside {deployment}") from None
    if not rel.parts:
        raise PackagingError("the package directory must be a sub-directory of deployment/")
    safe_path(Path(repo_root), PurePosixPath("deployment", *rel.parts).as_posix(), must_exist=False)
    return dest


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_checksums(repo_root: Path) -> dict[str, str]:
    """{allow-listed path: SHA-256} from the pipeline manifests; every allow-listed file must be covered."""
    recorded: dict[str, str] = {}
    for rel in SOURCE_MANIFESTS:
        with open(safe_path(repo_root, rel), encoding="utf-8") as fh:
            for entry in json.load(fh).get("files", []):
                if entry.get("path") in REQUIRED_FILES:
                    recorded[entry["path"]] = entry["sha256"]
    missing = [f for f in REQUIRED_FILES if f not in recorded]
    if missing:
        raise PackagingError(f"no recorded checksum in the pipeline manifests for: {', '.join(missing)}")
    return recorded


def recorded_fingerprint(repo_root: Path) -> str:
    with open(safe_path(repo_root, GRAPH_SNAPSHOT), encoding="utf-8") as fh:
        return json.load(fh)["fingerprint"]


def kg_fingerprint(kg_dir: Path) -> str:
    """The project's own fingerprint of the training graph view built from a directory of KG CSVs."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.kg.graph_view import load_view_from_kg_csvs

    return load_view_from_kg_csvs(kg_dir).fingerprint()


def _check_fingerprint(repo_root: Path, kg_dir: Path, fingerprint_fn) -> str:
    expected, actual = recorded_fingerprint(repo_root), fingerprint_fn(kg_dir)
    if actual != expected:
        raise PackagingError(f"KG fingerprint {actual} does not match the frozen Day 7 snapshot {expected} "
                             f"({GRAPH_SNAPSHOT}); the KG files are not the ones the recommender was frozen with")
    return actual


def verify_sources(repo_root: Path = REPO_ROOT, fingerprint_fn=kg_fingerprint) -> dict[str, dict]:
    """Check the allow-listed source files in the repository. Returns {path: {sha256, bytes}}."""
    check_allow_list()
    paths = {rel: safe_path(repo_root, rel) for rel in REQUIRED_FILES}  # all present before hashing anything
    recorded = recorded_checksums(repo_root)
    info = {}
    for rel, path in paths.items():
        digest = sha256_of(path)
        if digest != recorded[rel]:
            raise PackagingError(f"checksum mismatch for {rel}: the file differs from the pipeline manifest")
        info[rel] = {"sha256": digest, "bytes": path.stat().st_size}
    _check_fingerprint(repo_root, Path(repo_root) / KG_DIR, fingerprint_fn)
    return info


def verify_package(repo_root: Path = REPO_ROOT, dest: Path | None = None, fingerprint_fn=kg_fingerprint) -> dict[str, dict]:
    """Check an existing package without changing anything. Returns {path: {sha256, bytes}}."""
    check_allow_list()
    dest = _package_dir(repo_root, dest or Path(repo_root) / PACKAGE_DIR)
    if not dest.is_dir():
        raise PackagingError(f"no package at {dest} (create it first, without --verify)")
    found = set()
    for current, dirs, files in os.walk(dest, followlinks=False):
        for name in dirs + files:
            entry = Path(current) / name
            rel = entry.relative_to(dest).as_posix()
            if _is_link(entry):
                raise PackagingError(f"the package contains a symlink or junction: {rel}")
            if entry.is_file():
                found.add(rel)
    expected = set(REQUIRED_FILES) | {PACKAGE_MANIFEST}
    leaked = sorted(f for f in found if f.startswith(FORBIDDEN_PREFIXES) or any(p in f.split("/") for p in FORBIDDEN_PARTS))
    if leaked:
        raise PackagingError(f"the package contains private or raw data: {', '.join(leaked)}")
    if found != expected:
        details = []
        if expected - found:
            details.append("missing: " + ", ".join(sorted(expected - found)))
        if found - expected:
            details.append("unexpected: " + ", ".join(sorted(found - expected)))
        raise PackagingError("the package does not hold exactly the allow-listed files (" + "; ".join(details) + ")")
    with open(dest / PACKAGE_MANIFEST, encoding="utf-8") as fh:
        manifest = {e["path"]: e for e in json.load(fh)["files"]}
    if set(manifest) != set(REQUIRED_FILES):
        raise PackagingError(f"{PACKAGE_MANIFEST} does not list exactly the allow-listed files")
    # The pipeline manifests are the authority when they are available (they are, in the repository).
    authority = recorded_checksums(repo_root) if all((Path(repo_root) / m).is_file() for m in SOURCE_MANIFESTS) else None
    info = {}
    for rel in REQUIRED_FILES:
        digest = sha256_of(dest / rel)
        if digest != manifest[rel]["sha256"] or (authority is not None and digest != authority[rel]):
            raise PackagingError(f"checksum mismatch for packaged {rel}")
        info[rel] = {"sha256": digest, "bytes": (dest / rel).stat().st_size}
    _check_fingerprint(repo_root, dest / KG_DIR, fingerprint_fn)
    return info


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
def build_package(repo_root: Path = REPO_ROOT, dest: Path | None = None, *, force: bool = False,
                  fingerprint_fn=kg_fingerprint) -> dict[str, dict]:
    """Verify the sources, copy the allow-listed files into a fresh staging directory, verify the copy, and only
    then put it in place. An existing package is replaced only with force=True."""
    dest = _package_dir(repo_root, dest or Path(repo_root) / PACKAGE_DIR)
    if dest.exists() and not force:
        raise PackagingError(f"{dest} already exists; pass --force to replace it (or --verify to check it)")
    info = verify_sources(repo_root, fingerprint_fn)
    staging = dest.with_name(f".{dest.name}.staging-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        for rel in REQUIRED_FILES:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(repo_root, rel), target, follow_symlinks=False)
        manifest = {"description": "HireWise runtime data package (scripts/prepare_deployment_data.py)",
                    "kg_fingerprint": recorded_fingerprint(repo_root),
                    "files": [{"path": rel, **info[rel]} for rel in REQUIRED_FILES]}
        (staging / PACKAGE_MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        verify_package(repo_root, staging, fingerprint_fn)
        if dest.exists():  # force=True: replace the previous package only after the new one verified
            if _is_link(dest) or not dest.is_dir():
                raise PackagingError(f"refusing to replace {dest}: not a regular directory")
            shutil.rmtree(dest)
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return info


def _report(info: dict[str, dict], heading: str) -> None:
    print(heading)
    for rel in REQUIRED_FILES:
        print(f"  {info[rel]['bytes'] / 1e6:8.2f} MB  {rel}")
    total = sum(v["bytes"] for v in info.values())
    print(f"  {len(info)} files, {total / 1e6:.1f} MB ({total / 2**20:.1f} MiB) in total")


def main(argv=None, repo_root: Path = REPO_ROOT, dest: Path | None = None, fingerprint_fn=kg_fingerprint) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="check an existing package; copy nothing")
    mode.add_argument("--force", action="store_true", help="replace an existing package")
    args = parser.parse_args(argv)
    dest = dest or Path(repo_root) / PACKAGE_DIR
    try:
        if args.verify:
            info = verify_package(repo_root, dest, fingerprint_fn)
            _report(info, f"package verified: {dest}")
        else:
            info = build_package(repo_root, dest, force=args.force, fingerprint_fn=fingerprint_fn)
            _report(info, f"package created and verified: {dest}")
    except PackagingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
