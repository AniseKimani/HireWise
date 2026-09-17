"""Day 2 reproducible Upwork data preparation pipeline.

Usage (from repository root):

    python scripts/run_data_pipeline.py
    python scripts/run_data_pipeline.py --raw data/raw/upwork-jobs.csv --out data/processed

Stages: load config -> load raw data -> parse -> clean/flag -> build
vocabulary -> finalize eligibility -> sample requests -> background corpus
-> statistics -> save outputs -> print summary. Fails clearly if the raw
CSV is missing; never downloads data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import stats
from src.data.clean import parse_and_flag, finalize_flags
from src.data.config import DataConfig, load_config
from src.data.io import build_manifest, load_raw, save_csv, save_json, save_manifest
from src.data.sample import background_corpus, sample_requests
from src.data.vocab import build_vocabulary

LIST_JOIN = "|"

AUDIT_COLUMNS = [
    "link",
    "title",
    "published_date",
    "is_hourly",
    "hourly_low",
    "hourly_high",
    "budget",
    "country",
    "country_clean",
    "footer_found",
    "category",
    "n_skills_raw",
    "skills_raw",
    "skills_vocab",
    "location_requirement",
    "footer_country",
    "pay_type",
    "pay_amount",
    "price_tier",
    "is_duplicate",
    "has_no_skills",
    "is_in_scope_category",
    "has_vocab_skill",
    "exclusion_reason",
    "exclude_from_model",
    "eligible_for_model",
    "is_sampled_request",
]


def _serialize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("skills_raw", "skills_vocab"):
        out[col] = out[col].map(lambda skills: LIST_JOIN.join(skills))
    return out


def _save_audit_csv(df: pd.DataFrame, path: Path) -> Path:
    ordered = df.loc[:, AUDIT_COLUMNS].sort_values("link").reset_index(drop=True)
    return save_csv(_serialize_for_csv(ordered), path)


def _make_figures(df: pd.DataFrame, eligible: pd.Series, config: DataConfig, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    cat_freq = stats.category_frequencies(df, eligible).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(cat_freq["category"][::-1], cat_freq["count"][::-1])
    ax.set_xlabel("Eligible jobs")
    ax.set_title("Top 20 categories by job count (eligible modelling population)")
    fig.tight_layout()
    path = out_dir / "category_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    skill_counts = df.loc[eligible, "skills_vocab"].map(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(skill_counts, bins=range(0, int(skill_counts.max()) + 2), edgecolor="white")
    ax.set_xlabel("Controlled-vocabulary skills per job")
    ax.set_ylabel("Job count")
    ax.set_title("Skills per job (eligible modelling population)")
    fig.tight_layout()
    path = out_dir / "skills_per_job.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fixed = df.loc[eligible & (df["pay_type"] == "fixed"), "pay_amount"].dropna()
    low, high = fixed.quantile([config.winsorize_low_percentile, config.winsorize_high_percentile])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fixed.clip(low, high), bins=40, edgecolor="white")
    ax.set_xlabel("Fixed budget, $ (winsorised at 1st/99th percentile for display only)")
    ax.set_ylabel("Job count")
    ax.set_title("Fixed-price budget distribution (eligible modelling population)")
    fig.tight_layout()
    path = out_dir / "pay_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def run(raw_path: Path, out_dir: Path, config_path: Path) -> dict:
    config = load_config(config_path)

    print(f"[1/10] Loading configuration from {config_path}")
    print(f"[2/10] Loading raw dataset from {raw_path}")
    raw = load_raw(raw_path)

    print("[3/10] Parsing job descriptions")
    print("[4/10] Cleaning and flagging (duplicates, zero-skill, category scope)")
    flagged = parse_and_flag(raw, config)

    print("[5/10] Building controlled skill vocabulary")
    vocab, vocab_table, rare_table = build_vocabulary(flagged, config)

    print("[6/10] Determining eligibility")
    finalized = finalize_flags(flagged, vocab, config)

    print(f"[7/10] Sampling {config.model_requests} modelling requests (seed={config.seed})")
    sampled = sample_requests(finalized, config)

    print("[8/10] Building background corpus")
    background = background_corpus(sampled)

    print("[9/10] Calculating statistics")
    eligible_mask = sampled["eligible_for_model"]
    background_mask = sampled["eligible_for_model"] & ~sampled["is_sampled_request"]

    cat_stats = stats.category_stats(sampled, eligible_mask)
    skill_freq = stats.skill_frequencies(sampled, eligible_mask)
    pay_pct = stats.pay_percentiles(sampled, eligible_mask, config)
    location = stats.location_availability(sampled, eligible_mask)
    cooc_full = stats.skill_cooccurrence(sampled, eligible_mask, config)
    cooc_background = stats.skill_cooccurrence(sampled, background_mask, config)
    category_sim = stats.category_similarity(sampled, eligible_mask)

    print("[10/10] Saving outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    output_paths.append(_save_audit_csv(sampled, out_dir / "prepared_jobs.csv"))
    output_paths.append(_save_audit_csv(sampled.loc[eligible_mask], out_dir / "eligible_jobs.csv"))
    output_paths.append(_save_audit_csv(sampled.loc[sampled["is_sampled_request"]], out_dir / "sampled_requests.csv"))
    output_paths.append(_save_audit_csv(background, out_dir / "background_corpus.csv"))

    vocab_out = vocab_table.assign(in_vocabulary=True)
    rare_out = rare_table.assign(in_vocabulary=False)
    output_paths.append(save_csv(vocab_out, out_dir / "skills_vocab.csv"))
    output_paths.append(save_csv(rare_out, out_dir / "rare_skills.csv"))

    output_paths.append(save_csv(cat_stats, out_dir / "category_stats.csv"))
    output_paths.append(save_csv(skill_freq, out_dir / "skill_frequencies.csv"))
    output_paths.append(save_csv(cooc_full, out_dir / "skill_cooccurrence_full_corpus.csv"))
    output_paths.append(save_csv(cooc_background, out_dir / "skill_cooccurrence_background_corpus.csv"))
    output_paths.append(save_csv(category_sim, out_dir / "category_similarity.csv"))

    output_paths.append(save_json(pay_pct, out_dir / "pay_percentiles.json"))
    output_paths.append(save_json(location, out_dir / "location_availability.json"))

    figures_dir = out_dir / "figures"
    output_paths.extend(_make_figures(sampled, eligible_mask, config, figures_dir))

    n_raw = len(raw)
    n_dup = int(sampled["is_duplicate"].sum())
    n_zero_skill = int(sampled["has_no_skills"].sum())
    n_in_scope_categories = int(sampled.loc[sampled["is_in_scope_category"], "category"].nunique())
    n_dedup_devskill = int((~sampled["is_duplicate"] & ~sampled["has_no_skills"]).sum())
    n_in_scope_rows = int(
        (~sampled["is_duplicate"] & ~sampled["has_no_skills"] & sampled["is_in_scope_category"]).sum()
    )
    retention_pct = round(n_in_scope_rows / n_dedup_devskill * 100, 2) if n_dedup_devskill else 0.0

    summary = {
        "raw_rows": n_raw,
        "duplicate_rows": n_dup,
        "zero_skill_rows": n_zero_skill,
        "in_scope_categories": n_in_scope_categories,
        "in_scope_rows": n_in_scope_rows,
        "scope_retention_pct": retention_pct,
        "controlled_vocabulary_size": len(vocab),
        "eligible_modelling_jobs": int(eligible_mask.sum()),
        "sampled_requests": int(sampled["is_sampled_request"].sum()),
        "background_jobs": int(background_mask.sum()),
        "seed": config.seed,
    }
    summary_path = save_json(summary, out_dir / "summary_stats.json")
    output_paths.append(summary_path)

    manifest = build_manifest(REPO_ROOT, output_paths)
    save_manifest(manifest, out_dir / "manifest.json")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(REPO_ROOT / "data" / "raw" / "upwork-jobs.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "data_config.yaml"))
    args = parser.parse_args()

    summary = run(Path(args.raw), Path(args.out), Path(args.config))

    print("\n=== Day 2 pipeline summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
