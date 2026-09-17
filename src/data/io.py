"""Raw dataset loading and reproducible output writing.

Reproducibility rules (see docs/experimental_design.md Section 16):

- The raw CSV is treated as read-only and is never written to.
- CSV outputs use UTF-8, LF line endings, and stable row/column ordering
  so that re-running the pipeline on the same inputs produces
  byte-identical files.
- The manifest records relative (not absolute/OS-dependent) paths and a
  content hash per output file, with no embedded timestamps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED_RAW_COLUMNS = [
    "title",
    "link",
    "description",
    "published_date",
    "is_hourly",
    "hourly_low",
    "hourly_high",
    "budget",
    "country",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw Upwork CSV. Fails clearly if the file is missing;
    never downloads or fabricates data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place the Upwork job-postings "
            "CSV there (it is git-ignored and must be provided locally); "
            "this pipeline does not download data automatically."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}")
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame deterministically: UTF-8, LF line endings, no
    index column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: str | Path, output_paths: list[str | Path]) -> dict:
    """Build a manifest of relative paths, row counts (for CSVs), and
    SHA-256 hashes. Paths are stored as POSIX-style relative strings so the
    manifest is identical on Windows and Unix."""
    root = Path(root)
    entries = []
    for p in sorted(str(Path(p).relative_to(root).as_posix()) for p in output_paths):
        full = root / p
        entry = {"path": p, "sha256": sha256_of(full), "bytes": full.stat().st_size}
        if full.suffix == ".csv":
            # A naive newline count is wrong when a quoted field embeds a
            # literal newline (this happens in real job titles), so count
            # rows through the csv module instead.
            import csv

            with open(full, "r", encoding="utf-8", newline="") as fh:
                entry["rows"] = sum(1 for _ in csv.reader(fh)) - 1
        entries.append(entry)
    return {"files": entries}


def save_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path
