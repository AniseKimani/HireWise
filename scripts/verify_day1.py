"""Day 1 verification and cleaning-decision analysis for the Upwork Job Postings dataset.

Re-derives every number cited in docs/experimental_design.md from the raw CSV.
Read-only: the raw file is never modified.

Usage:
    python scripts/verify_day1.py data/raw/upwork-jobs.csv docs/day1_stats.json
"""
import html
import json
import re
import sys
from collections import Counter

import pandas as pd

RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/upwork-jobs.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "docs/day1_stats.json"
STOPS = ["Skills:", "Country:", "Location Requirement:", "click to apply"]
PAY_COLS = ["is_hourly", "hourly_low", "hourly_high", "budget", "country"]


def footer(text: str) -> str:
    """Platform metadata footer: text from the LAST 'Posted On:' marker.
    Anchoring here avoids false matches when clients write 'Category:' in the job body."""
    i = text.rfind("Posted On:")
    return text[i:] if i >= 0 else ""


def field(foot: str, name: str, stops=STOPS):
    pattern = rf"{name}:\s*(.*?)(?={'|'.join(map(re.escape, stops))}|$)"
    m = re.search(pattern, foot, re.S)
    return m.group(1).strip() if m else None


def parse_skills(foot: str) -> list:
    block = field(foot, "Skills") or ""  # first Skills block only (the footer repeats it)
    cleaned = (re.sub(r"\s+", " ", s).strip() for s in block.split(","))
    return list(dict.fromkeys(s for s in cleaned if s))  # dedupe within job, keep order


def quantiles(s: pd.Series) -> dict:
    return {str(k): round(float(v), 2) for k, v in s.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).items()}


raw = pd.read_csv(RAW_PATH)
S = {"raw": {"rows": len(raw), "columns": raw.shape[1], "column_names": list(raw.columns)}}

foot = raw["description"].map(html.unescape).map(footer)
raw["category"] = foot.map(lambda f: field(f, "Category"))
raw["skills"] = foot.map(parse_skills)
raw["location_req"] = foot.map(lambda f: field(f, "Location Requirement", ["Country:", "click to apply"]))
ts = pd.to_datetime(raw["published_date"], utc=True)

# ---- 1. Raw facts -------------------------------------------------------------
dup_mask = raw.duplicated(["title", "description"], keep=False)
dups = raw[dup_mask]
skill_jobfreq = Counter(s for lst in raw["skills"] for s in lst)
S["raw"].update({
    "footer_found": int((foot != "").sum()),
    "dup_links": int(raw["link"].duplicated().sum()),
    "dup_title_desc_groups": int(dups.groupby(["title", "description"]).ngroups),
    "dup_title_desc_rows_removed": int(raw.duplicated(["title", "description"]).sum()),
    "dup_groups_differing_in_pay_or_country": int(sum(dups.groupby(["title", "description"])[c].nunique(dropna=False).gt(1).sum() for c in PAY_COLS)),
    "dup_max_minutes_apart": round(float(dups.assign(t=ts[dup_mask]).groupby(["title", "description"])["t"].agg(lambda x: (x.max() - x.min()).total_seconds() / 60).max()), 1),
    "zero_skill_jobs": int((raw["skills"].map(len) == 0).sum()),
    "categories": int(raw["category"].nunique()),
    "missing_category": int(raw["category"].isna().sum()),
    "unique_skills": len(skill_jobfreq),
    "case_insensitive_skill_collisions": sum(v > 1 for v in Counter(s.casefold() for s in skill_jobfreq).values()),
    "skills_jobfreq_ge20": sum(v >= 20 for v in skill_jobfreq.values()),
    "date_min": str(ts.min()), "date_max": str(ts.max()),
    "share_posted_2024_02_12_onwards_pct": round(float((ts >= "2024-02-12").mean() * 100), 2),
    "jobs_per_day_from_2024_02_10": {str(k): int(v) for k, v in ts[ts >= "2024-02-10"].dt.date.value_counts().sort_index().items()},
    "country_missing": int(raw["country"].isna().sum()),
    "country_unique": int(raw["country"].nunique()),
    "country_top10_pct_of_all_rows": (raw["country"].value_counts().head(10) / len(raw) * 100).round(2).to_dict(),
    "pay_type": raw["is_hourly"].astype("string").map({"True": "hourly", "False": "fixed"}).fillna("unspecified").value_counts().to_dict(),
    "budget_quantiles": quantiles(raw["budget"].dropna()),
    "hourly_low_quantiles": quantiles(raw["hourly_low"].dropna()),
    "hourly_high_quantiles": quantiles(raw["hourly_high"].dropna()),
    "hourly_missing_high": int(((raw["is_hourly"].astype("string") == "True") & raw["hourly_high"].isna()).sum()),
    "budget_gt_50000": int((raw["budget"] > 50000).sum()),
    "location_requirements": raw["location_req"].value_counts().to_dict(),
})

# ---- 2. Cleaning-decision analysis (proposed order) ----------------------------
raw["_completeness"] = raw[PAY_COLS].notna().sum(axis=1)
dedup = raw.sort_values(["_completeness", "link"], ascending=[False, True]).drop_duplicates(["title", "description"])
dev = dedup[dedup["skills"].map(len) > 0]
zero_skill = dedup[dedup["skills"].map(len) == 0]
S["cleaning"] = {
    "after_dedup": len(dedup),
    "after_zero_skill_removal": len(dev),
    "zero_skill_top_categories": zero_skill["category"].value_counts().head(5).to_dict(),
}

cat_counts = dev["category"].value_counts()
S["category_thresholds"] = {
    str(t): {"categories_kept": int((cat_counts >= t).sum()),
             "categories_removed_pct": round((1 - (cat_counts >= t).mean()) * 100, 1),
             "jobs_retained": int(cat_counts[cat_counts >= t].sum()),
             "jobs_retained_pct": round(cat_counts[cat_counts >= t].sum() / len(dev) * 100, 2)}
    for t in [20, 50, 100, 200]
}
scope = dev[dev["category"].isin(cat_counts[cat_counts >= 100].index)]


def vocab_table(frame: pd.DataFrame) -> dict:
    jf = Counter(s for lst in frame["skills"] for s in lst)
    total = sum(jf.values())
    rows = {}
    for t in [5, 10, 20, 50]:
        vocab = {s for s, c in jf.items() if c >= t}
        n_vocab = frame["skills"].map(lambda l: sum(s in vocab for s in l))
        rows[str(t)] = {"vocab_size": len(vocab),
                        "mentions_kept_pct": round(sum(c for s, c in jf.items() if c >= t) / total * 100, 2),
                        "jobs_with_0_vocab_skills": int((n_vocab == 0).sum()),
                        "mean_vocab_skills_per_job": round(float(n_vocab.mean()), 2)}
    return {"unique_skills": len(jf), "thresholds": rows}


S["vocab_on_dev_set"] = vocab_table(dev)
S["vocab_on_in_scope_set"] = vocab_table(scope)

# Evidence against fuzzy merging: naive plural stripping produces false merges.
jf_scope = Counter(s for lst in scope["skills"] for s in lst)
vocab20 = {s for s, c in jf_scope.items() if c >= 20}
S["naive_plural_merge_examples"] = [(s, v) for s in jf_scope if s not in vocab20 for v in vocab20
                                    if s != v and s.lower().rstrip("s") == v.lower().rstrip("s")]

eligible = scope[scope["skills"].map(lambda l: any(s in vocab20 for s in l))]
winsor = {}
for col in ["budget", "hourly_low", "hourly_high"]:
    s = eligible[col].dropna()
    p1, p99 = s.quantile([.01, .99])
    winsor[col] = {"n": len(s), "p1": round(float(p1), 2), "p99": round(float(p99), 2),
                   "n_above_p99": int((s > p99).sum()), "mean_stored": round(float(s.mean()), 2),
                   "mean_analysis_winsorized": round(float(s.clip(p1, p99).mean()), 2), "median": float(s.median())}
S["eligible_request_pool"] = {
    "rows": len(eligible),
    "categories": int(eligible["category"].nunique()),
    "pay_type": eligible["is_hourly"].astype("string").map({"True": "hourly", "False": "fixed"}).fillna("unspecified").value_counts().to_dict(),
    "location_requirements": eligible["location_req"].value_counts().to_dict(),
    "country_missing": int(eligible["country"].isna().sum()),
    "largest_category_share_pct": round(float(eligible["category"].value_counts(normalize=True).iloc[0] * 100), 2),
    "smallest_category_share_pct": round(float(eligible["category"].value_counts(normalize=True).iloc[-1] * 100), 3),
    "pay_distributions_winsorized_for_analysis": winsor,
}

with open(OUT_PATH, "w") as fh:
    json.dump(S, fh, indent=2, default=str)
print(json.dumps(S, indent=2, default=str))
