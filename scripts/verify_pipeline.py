"""Independent verification of Day 2 pipeline invariants.

This script re-derives every checked value from the raw CSV and from the
pipeline's own output files, using logic separate from
src/data/*.py where practical (e.g. duplicate/zero-skill/category counts
are recomputed with plain pandas here, not by calling clean.py), so it is
a genuine independent check rather than a restatement of numbers the
pipeline already produced.

Usage:
    python scripts/verify_pipeline.py [--raw PATH] [--processed PATH] [--config PATH]

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.data.config import load_config

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def independent_footer(text: str) -> str:
    decoded = html.unescape(text)
    idx = decoded.rfind("Posted On:")
    return decoded[idx:] if idx >= 0 else ""


def independent_skills(footer: str) -> list[str]:
    stops = ["Skills:", "Location Requirement:", "Country:", "click to apply"]
    pattern = rf"Skills:\s*(.*?)(?:{'|'.join(re.escape(s) for s in stops)}|$)"
    m = re.search(pattern, footer, re.S)
    if not m:
        return []
    block = m.group(1)
    parts = re.split(r",\s{2,}", block)
    cleaned = (re.sub(r"\s+", " ", p).strip() for p in parts)
    return list(dict.fromkeys(p for p in cleaned if p))


def independent_category(footer: str) -> str | None:
    m = re.search(r"Category:\s*(.*?)(?:Skills:|Location Requirement:|Country:|click to apply|$)", footer, re.S)
    return m.group(1).strip() if m and m.group(1).strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "raw" / "upwork-jobs.csv"))
    parser.add_argument("--processed", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "data_config.yaml"))
    args = parser.parse_args()

    raw_path = Path(args.raw)
    processed = Path(args.processed)
    config = load_config(args.config)

    if not raw_path.exists():
        print(f"Raw dataset not found at {raw_path}; cannot verify.")
        return 1

    raw_before_mtime = raw_path.stat().st_mtime
    raw_before_size = raw_path.stat().st_size

    raw = pd.read_csv(raw_path)

    # ---- Independent recomputation from the raw CSV -----------------------
    footers = raw["description"].map(independent_footer)
    categories = footers.map(independent_category)
    skills = footers.map(independent_skills)

    check("raw row count == 53,058", len(raw) == 53058, f"got {len(raw)}")
    check("no duplicate links", raw["link"].duplicated().sum() == 0)

    completeness_cols = ["is_hourly", "hourly_low", "hourly_high", "budget", "country"]
    completeness = raw[completeness_cols].notna().sum(axis=1)
    order = pd.DataFrame({"completeness": completeness, "link": raw["link"], "category": categories,
                          "n_skills": skills.map(len)}, index=raw.index)
    dup_mask = raw.duplicated(["title", "description"], keep=False)
    n_dup_groups = raw[dup_mask].groupby(["title", "description"]).ngroups
    check("duplicate title+description groups == 22", n_dup_groups == 22, f"got {n_dup_groups}")

    ranked = order.sort_values(["completeness", "link"], ascending=[False, True])
    dup_key = raw.loc[ranked.index, ["title", "description"]]
    keep_mask = ~dup_key.duplicated(keep="first")
    is_duplicate = pd.Series(True, index=raw.index)
    is_duplicate.loc[ranked.index[keep_mask.values]] = False
    check("independently recomputed duplicate rows == 22", int(is_duplicate.sum()) == 22, f"got {int(is_duplicate.sum())}")

    has_no_skills = order["n_skills"] == 0
    n_zero_skill = int((has_no_skills & ~is_duplicate).sum()) + 0
    n_zero_skill_all_dup_rows_included = int(has_no_skills.sum())
    check("zero-skill jobs (pre-dedup) == 515", n_zero_skill_all_dup_rows_included == 515,
          f"got {n_zero_skill_all_dup_rows_included}")

    dev = order.loc[~is_duplicate & ~has_no_skills]
    cat_counts = dev["category"].value_counts()
    in_scope_categories = set(cat_counts[cat_counts >= config.min_category_jobs].index)
    check("in-scope categories == 88", len(in_scope_categories) == 88, f"got {len(in_scope_categories)}")

    in_scope_rows = dev["category"].isin(in_scope_categories).sum()
    retention = round(in_scope_rows / len(dev) * 100, 2)
    check("scope retention == 92.34%", abs(retention - 92.34) < 0.01, f"got {retention}")

    scope = dev.loc[dev["category"].isin(in_scope_categories)]
    scope_links = set(scope["link"])
    from collections import Counter
    scope_skill_lists = skills.loc[scope.index]
    freq = Counter(s for lst in scope_skill_lists for s in lst)
    vocab = {s for s, c in freq.items() if c >= config.min_skill_frequency}
    check("controlled vocabulary size == 1,187", len(vocab) == 1187, f"got {len(vocab)}")

    # ---- Cross-check against the pipeline's own output files --------------
    required_files = [
        "prepared_jobs.csv", "eligible_jobs.csv", "sampled_requests.csv",
        "background_corpus.csv", "skills_vocab.csv", "rare_skills.csv",
        "summary_stats.json", "manifest.json",
    ]
    missing = [f for f in required_files if not (processed / f).exists()]
    check("all expected output files exist", not missing, f"missing {missing}")
    if missing:
        print("\nRun scripts/run_data_pipeline.py first.")
        return 1

    summary = json.loads((processed / "summary_stats.json").read_text(encoding="utf-8"))
    check("pipeline summary raw_rows == 53,058", summary["raw_rows"] == 53058)
    check("pipeline summary duplicate_rows == 22", summary["duplicate_rows"] == 22)
    check("pipeline summary zero_skill_rows == 515", summary["zero_skill_rows"] == 515)
    check("pipeline summary in_scope_categories == 88", summary["in_scope_categories"] == 88)
    check("pipeline summary controlled_vocabulary_size == 1,187", summary["controlled_vocabulary_size"] == 1187)
    check("pipeline summary sampled_requests == config.model_requests",
          summary["sampled_requests"] == config.model_requests)
    check("pipeline seed matches config.seed", summary["seed"] == config.seed)

    requests_df = pd.read_csv(processed / "sampled_requests.csv")
    background_df = pd.read_csv(processed / "background_corpus.csv")
    eligible_df = pd.read_csv(processed / "eligible_jobs.csv")

    check("request count == config.model_requests", len(requests_df) == config.model_requests,
          f"got {len(requests_df)}")
    check("request/background link sets are disjoint",
          set(requests_df["link"]).isdisjoint(set(background_df["link"])))
    check("requests + background == eligible population",
          len(requests_df) + len(background_df) == len(eligible_df),
          f"{len(requests_df)} + {len(background_df)} != {len(eligible_df)}")
    check("no Dataset 2 / second dataset file referenced in outputs",
          not any("dataset2" in f.name.lower() or "dataset_2" in f.name.lower() for f in processed.glob("*")))

    required_flag_columns = {
        "is_duplicate", "has_no_skills", "is_in_scope_category", "has_vocab_skill",
        "exclude_from_model", "exclusion_reason", "eligible_for_model", "is_sampled_request",
    }
    prepared_columns = set(pd.read_csv(processed / "prepared_jobs.csv", nrows=1).columns)
    check("prepared_jobs.csv carries all required audit flags",
          required_flag_columns.issubset(prepared_columns),
          f"missing {required_flag_columns - prepared_columns}")

    # ---- Determinism: sampling must be stable across runs -----------------
    from src.data.clean import finalize_flags, parse_and_flag
    from src.data.sample import sample_requests
    from src.data.vocab import build_vocabulary

    flagged = parse_and_flag(raw, config)
    built_vocab, _, _ = build_vocabulary(flagged, config)
    finalized = finalize_flags(flagged, built_vocab, config)
    run_a = sample_requests(finalized, config)["is_sampled_request"]
    run_b = sample_requests(finalized, config)["is_sampled_request"]
    check("deterministic sampling: two runs agree exactly", (run_a.values == run_b.values).all())

    # ---- Config consistency ------------------------------------------------
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    check("config split ratios sum to 1.0", abs(ratio_sum - 1.0) < 1e-9, f"got {ratio_sum}")

    # ---- Raw file was never modified --------------------------------------
    check("raw file mtime unchanged during verification", raw_path.stat().st_mtime == raw_before_mtime)
    check("raw file size unchanged during verification", raw_path.stat().st_size == raw_before_size)

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
