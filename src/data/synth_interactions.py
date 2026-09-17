"""Multi-factor latent-utility interaction simulator
(docs/experimental_design.md Section 9; Day 3 brief Sections 9-13).

Two-pass design [IMPL]:

  Pass 1 (moment estimation, `estimate_utility_moments`): for every
  request, in any order, build the eligible pool *ignoring* worker
  hire-capacity -- capacity depends on hire outcomes, which depend on
  standardized utility, which the approved design defines as standardized
  "across all generated request-applicant pairs". Computing that live
  during a capacity-aware chronological pass is circular. Pass 1 samples
  applicants with the same raw skill-fit weighting used in the real pass
  and accumulates their raw utility components (S, T, Q, B, C) to obtain
  global per-component mean/std, the realised signal variance (used to
  calibrate the noise term to the configured variance share), and a
  hire-probability threshold calibrated to the target hire rate.

  Pass 2 (`simulate`): processes requests in chronological order of
  simulated day, with real hire-capacity tracking. Each pair's raw
  components are standardized using Pass 1's frozen moments, combined
  with the configured weights and a fresh noise draw, then shortlisted
  and (possibly) hired.

Variable classification enforced by this module's return values:
  - Model-visible: `interactions_df` (link, worker_id, client_id,
    simulated_day, event, rank_in_shortlist).
  - Generator-private truth (must never be merged into model-visible
    tables): `truth_df` (raw and standardized S/T/Q/B/C, noise, utility,
    hired flag) -- see docs/synthetic_data_report.md's leakage section.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.data.config import DataConfig
from src.data.generator_config import GeneratorConfig
from src.data.synth_rng import stage_rng
from src.data.synth_workers import extract_required_countries

CANDIDATE_POOL_CAP = 150
COMPONENT_KEYS = ("S", "T", "Q", "B", "C")
_WEIGHT_KEY_BY_COMPONENT = {"S": "skill", "T": "taste", "Q": "quality", "B": "budget", "C": "category"}


# ---------------------------------------------------------------------------
# Skill relatedness and proficiency lookups
# ---------------------------------------------------------------------------

def build_skill_relatedness(cooccurrence_df: pd.DataFrame, topn: int) -> dict[str, list[tuple[str, float]]]:
    """Undirected top-`topn` NPMI neighbours per skill, from the
    (skill_a < skill_b) pairwise table produced by src/data/stats.py."""
    neighbours: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in cooccurrence_df.itertuples():
        neighbours[row.skill_a].append((row.skill_b, row.npmi))
        neighbours[row.skill_b].append((row.skill_a, row.npmi))
    return {skill: sorted(pairs, key=lambda p: -p[1])[:topn] for skill, pairs in neighbours.items()}


def build_proficiency_matrix(
    workers_private_df: pd.DataFrame, vocab_skills: list[str]
) -> tuple[np.ndarray, dict[str, int]]:
    skill_index = {s: i for i, s in enumerate(vocab_skills)}
    matrix = np.zeros((len(workers_private_df), len(vocab_skills)), dtype=np.float32)
    for row_i, (true_skills, proficiency) in enumerate(
        zip(workers_private_df["true_skills"], workers_private_df["proficiency"])
    ):
        if not isinstance(true_skills, str) or not true_skills:
            continue
        skills = true_skills.split("|")
        profs = [float(p) for p in proficiency.split("|")]
        for skill, prof in zip(skills, profs):
            idx = skill_index.get(skill)
            if idx is not None:
                matrix[row_i, idx] = prof
    return matrix, skill_index


def build_neighbor_indices(
    relatedness: dict[str, list[tuple[str, float]]], skill_index: dict[str, int]
) -> dict[int, np.ndarray]:
    result = {}
    for skill, idx in skill_index.items():
        neighbour_idx = [skill_index[s] for s, _ in relatedness.get(skill, []) if s in skill_index]
        if neighbour_idx:
            result[idx] = np.array(neighbour_idx, dtype=int)
    return result


def compute_skill_fit(
    worker_positions: np.ndarray,
    request_skill_idx: list[int],
    proficiency_matrix: np.ndarray,
    neighbor_indices: dict[int, np.ndarray],
    lam: float,
) -> np.ndarray:
    """S(r, w): mean over the request's required vocabulary skills of
    max(p_w(s), lam * max_{s' in N(s)} p_w(s'))."""
    if not request_skill_idx or len(worker_positions) == 0:
        return np.zeros(len(worker_positions))

    sub = proficiency_matrix[worker_positions]
    components = np.zeros((len(worker_positions), len(request_skill_idx)))
    for j, skill_idx in enumerate(request_skill_idx):
        direct = sub[:, skill_idx]
        neighbours = neighbor_indices.get(skill_idx)
        neighbour_max = sub[:, neighbours].max(axis=1) if neighbours is not None else np.zeros(len(worker_positions))
        components[:, j] = np.maximum(direct, lam * neighbour_max)
    return components.mean(axis=1)


# ---------------------------------------------------------------------------
# Candidate eligibility
# ---------------------------------------------------------------------------

class CandidateIndex:
    """Precomputed per-category worker positions so eligible-pool
    construction is O(pool size), not O(all workers)."""

    def __init__(self, workers_df: pd.DataFrame, top_country_names: list[str]):
        self.workers_df = workers_df.reset_index(drop=True)
        self.join_day = self.workers_df["join_day"].to_numpy()
        self.country = self.workers_df["country"].to_numpy()
        self.top_country_names = top_country_names

        primary = defaultdict(list)
        secondary = defaultdict(list)
        for pos, row in enumerate(self.workers_df.itertuples()):
            primary[row.primary_category].append(pos)
            secs = row.secondary_categories.split("|") if isinstance(row.secondary_categories, str) and row.secondary_categories else []
            for cat in secs:
                secondary[cat].append(pos)
        self.primary_positions = {k: np.array(v, dtype=int) for k, v in primary.items()}
        self.secondary_positions = {k: np.array(v, dtype=int) for k, v in secondary.items()}

    def _category_pool(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        primary = self.primary_positions.get(category, np.array([], dtype=int))
        secondary = self.secondary_positions.get(category, np.array([], dtype=int))
        secondary = np.setdiff1d(secondary, primary, assume_unique=False)
        pool = np.concatenate([primary, secondary])
        fit = np.concatenate([np.ones(len(primary)), np.full(len(secondary), 0.6)])
        return pool, fit

    def eligible_pool(
        self, category: str, day: float, location_requirement: object, blocked: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        pool, fit = self._category_pool(category)
        if len(pool) == 0:
            return pool, fit
        mask = self.join_day[pool] <= day
        required = extract_required_countries(location_requirement, self.top_country_names)
        if required:
            mask &= np.isin(self.country[pool], required)
        if blocked is not None and blocked.size:
            mask &= ~np.isin(pool, blocked)
        return pool[mask], fit[mask]


def _subsample(rng: np.random.Generator, pool: np.ndarray, fit: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pool) <= cap:
        return pool, fit
    keep = rng.choice(len(pool), size=cap, replace=False)
    return pool[keep], fit[keep]


def _sample_applicants(
    rng: np.random.Generator, pool: np.ndarray, skill_fit: np.ndarray, base_prob: float, max_n: int
) -> np.ndarray:
    if len(pool) == 0:
        return pool
    weights = np.clip(base_prob + skill_fit, 1e-6, None)
    weights = weights / weights.sum()
    n = min(max_n, len(pool))
    return rng.choice(pool, size=n, replace=False, p=weights)


def _raw_components(
    applicant_positions: np.ndarray,
    category_fit: np.ndarray,
    skill_fit: np.ndarray,
    workers_df: pd.DataFrame,
    workers_private: pd.DataFrame,
    request_price_tier: float,
    client_taste: np.ndarray,
    taste_cols: list[str],
) -> dict[str, np.ndarray]:
    worker_price_tier = workers_df["price_tier"].to_numpy()[applicant_positions]
    if pd.isna(request_price_tier):
        budget = np.full(len(applicant_positions), 0.7)
    else:
        diff = worker_price_tier - request_price_tier - 0.1
        budget = np.where(diff <= 0, 1.0, np.exp(-5.0 * diff))

    quality = workers_private["quality"].to_numpy()[applicant_positions]
    worker_taste = workers_private[taste_cols].to_numpy()[applicant_positions]
    taste = worker_taste @ client_taste / np.sqrt(len(taste_cols))

    return {"S": skill_fit, "T": taste, "Q": quality, "B": budget, "C": category_fit}


def _standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def _request_skill_indices(skills_vocab_cell: object, skill_index: dict[str, int]) -> list[int]:
    skills = skills_vocab_cell.split("|") if isinstance(skills_vocab_cell, str) and skills_vocab_cell else []
    return [skill_index[s] for s in skills if s in skill_index]


def candidate_pool_sizes(timeline_df: pd.DataFrame, requests_info: pd.DataFrame, candidate_index: CandidateIndex) -> pd.DataFrame:
    """Eligible-pool size per request, ignoring capacity (Day 3 brief
    Section 9's reporting requirement: mean/median/min/max candidates per
    request, and requests with very small pools)."""
    merged = timeline_df.merge(requests_info, on="link", how="left")
    sizes = []
    for row in merged.itertuples():
        pool, _ = candidate_index.eligible_pool(row.category, row.simulated_day, row.location_requirement, blocked=None)
        sizes.append(len(pool))
    return pd.DataFrame({"link": merged["link"].values, "n_candidates": sizes})


# ---------------------------------------------------------------------------
# Pass 1: moment estimation
# ---------------------------------------------------------------------------

def estimate_utility_moments(
    timeline_df: pd.DataFrame,
    requests_info: pd.DataFrame,
    clients_private: pd.DataFrame,
    workers_private: pd.DataFrame,
    candidate_index: CandidateIndex,
    proficiency_matrix: np.ndarray,
    skill_index: dict[str, int],
    neighbor_indices: dict[int, np.ndarray],
    config: DataConfig,
    gen_config: GeneratorConfig,
) -> dict:
    rng_elig = stage_rng(config.seed, "candidate_eligibility")
    rng_app = stage_rng(config.seed, "applicant_sampling")
    rng_noise = stage_rng(config.seed, "utility_noise")
    rng_shortlist = stage_rng(config.seed, "shortlist_calibration")

    merged = timeline_df.merge(requests_info, on="link", how="left")
    taste_lookup = clients_private.set_index("client_id")
    taste_cols = [c for c in clients_private.columns if c.startswith("taste_")]

    raw: dict[str, list[np.ndarray]] = {k: [] for k in COMPONENT_KEYS}
    shortlist_sizes: list[int] = []
    for row in merged.itertuples():
        req_skill_idx = _request_skill_indices(row.skills_vocab, skill_index)
        pool, cat_fit = candidate_index.eligible_pool(row.category, row.simulated_day, row.location_requirement, blocked=None)
        pool, cat_fit = _subsample(rng_elig, pool, cat_fit, CANDIDATE_POOL_CAP)
        if len(pool) == 0:
            continue

        skill_fit_pool = compute_skill_fit(pool, req_skill_idx, proficiency_matrix, neighbor_indices, gen_config.skill_relatedness_lambda)
        applicants = _sample_applicants(rng_app, pool, skill_fit_pool, gen_config.applicant_base_prob, gen_config.applicant_pool_max)
        if len(applicants) == 0:
            continue

        cat_fit_by_pos = dict(zip(pool.tolist(), cat_fit.tolist()))
        applicant_cat_fit = np.array([cat_fit_by_pos[p] for p in applicants])
        applicant_skill_fit = compute_skill_fit(applicants, req_skill_idx, proficiency_matrix, neighbor_indices, gen_config.skill_relatedness_lambda)

        client_taste = taste_lookup.loc[row.client_id, taste_cols].to_numpy(dtype=float)
        components = _raw_components(
            applicants, applicant_cat_fit, applicant_skill_fit, candidate_index.workers_df,
            workers_private, row.price_tier, client_taste, taste_cols,
        )
        for key in COMPONENT_KEYS:
            raw[key].append(components[key])
        shortlist_sizes.append(min(gen_config.shortlist_size, len(applicants)))

    moments = {}
    concatenated = {}
    for key in COMPONENT_KEYS:
        values = np.concatenate(raw[key]) if raw[key] else np.array([0.0])
        concatenated[key] = values
        moments[key] = (float(values.mean()), float(values.std()))

    weights = gen_config.utility_weights
    signal = sum(
        weights[_WEIGHT_KEY_BY_COMPONENT[key]] * _standardize(concatenated[key], *moments[key])
        for key in COMPONENT_KEYS
    )
    signal_var = float(signal.var()) if signal.size else 1.0

    share = gen_config.utility_noise_variance_share
    noise_std = float(np.sqrt(max(share, 1e-9) / max(1 - share, 1e-9) * signal_var))

    # Match pass 2's hire rule exactly: the hire decision looks at the max
    # utility *within the sampled shortlist*, not the full applicant pool,
    # so calibration must shortlist here too (see module docstring).
    max_u_per_request = []
    cursor = 0
    for arr, n_short in zip(raw["S"], shortlist_sizes):
        n = len(arr)
        pair_signal = signal[cursor:cursor + n]
        noise = rng_noise.normal(0, noise_std, size=n)
        utility = pair_signal + noise
        positions = np.arange(n)
        _, short_utility = _plackett_luce_sample(rng_shortlist, positions, utility, n_short, gen_config.shortlist_temperature)
        max_u_per_request.append(float(short_utility.max()))
        cursor += n
    max_u_per_request = np.array(max_u_per_request) if max_u_per_request else np.array([0.0])

    alpha = 2.0 / max(float(max_u_per_request.std()), 1e-6)
    theta = _calibrate_theta(max_u_per_request, alpha, gen_config.hire_rate_target)

    return {
        "moments": moments,
        "noise_std": noise_std,
        "alpha": alpha,
        "theta": theta,
        "signal_var": signal_var,
        "pass1_hire_rate_estimate": float(np.mean(1 / (1 + np.exp(-alpha * (max_u_per_request - theta))))),
    }


def _calibrate_theta(max_u_samples: np.ndarray, alpha: float, target_rate: float, iters: int = 60) -> float:
    lo, hi = float(max_u_samples.min()) - 5, float(max_u_samples.max()) + 5
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = float(np.mean(1 / (1 + np.exp(-alpha * (max_u_samples - mid)))))
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Pass 2: chronological simulation with real capacity tracking
# ---------------------------------------------------------------------------

class _CapacityTracker:
    def __init__(self, max_hires: int, window_days: float):
        self.max_hires = max_hires
        self.window_days = window_days
        self._hire_days: dict[int, deque] = defaultdict(deque)

    def blocked(self, day: float) -> np.ndarray:
        blocked = []
        for pos, days in self._hire_days.items():
            while days and day - days[0] > self.window_days:
                days.popleft()
            if len(days) >= self.max_hires:
                blocked.append(pos)
        return np.array(blocked, dtype=int)

    def record_hire(self, worker_pos: int, day: float) -> None:
        self._hire_days[worker_pos].append(day)


def simulate(
    timeline_df: pd.DataFrame,
    requests_info: pd.DataFrame,
    clients_private: pd.DataFrame,
    workers_private: pd.DataFrame,
    candidate_index: CandidateIndex,
    proficiency_matrix: np.ndarray,
    skill_index: dict[str, int],
    neighbor_indices: dict[int, np.ndarray],
    calibration: dict,
    config: DataConfig,
    gen_config: GeneratorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (interactions_df, truth_df). See module docstring for the
    model-visible / generator-private split."""
    rng_elig = stage_rng(config.seed, "candidate_eligibility")
    rng_app = stage_rng(config.seed, "applicant_sampling")
    rng_noise = stage_rng(config.seed, "utility_noise")
    rng_shortlist = stage_rng(config.seed, "shortlist")
    rng_hire = stage_rng(config.seed, "hire")

    merged = timeline_df.merge(requests_info, on="link", how="left").sort_values(["simulated_day", "link"])
    taste_lookup = clients_private.set_index("client_id")
    taste_cols = [c for c in clients_private.columns if c.startswith("taste_")]

    moments = calibration["moments"]
    weights = gen_config.utility_weights
    noise_std = calibration["noise_std"]
    alpha = calibration["alpha"]
    theta = calibration["theta"]

    tracker = _CapacityTracker(gen_config.worker_capacity_max_hires, gen_config.worker_capacity_window_days)

    interaction_rows = []
    truth_rows = []
    worker_ids = candidate_index.workers_df["worker_id"].to_numpy()

    for row in merged.itertuples():
        day = row.simulated_day
        blocked = tracker.blocked(day)
        req_skill_idx = _request_skill_indices(row.skills_vocab, skill_index)

        pool, cat_fit = candidate_index.eligible_pool(row.category, day, row.location_requirement, blocked=blocked)
        pool, cat_fit = _subsample(rng_elig, pool, cat_fit, CANDIDATE_POOL_CAP)
        if len(pool) == 0:
            continue

        skill_fit_pool = compute_skill_fit(pool, req_skill_idx, proficiency_matrix, neighbor_indices, gen_config.skill_relatedness_lambda)
        applicants = _sample_applicants(rng_app, pool, skill_fit_pool, gen_config.applicant_base_prob, gen_config.applicant_pool_max)
        if len(applicants) == 0:
            continue

        cat_fit_by_pos = dict(zip(pool.tolist(), cat_fit.tolist()))
        applicant_cat_fit = np.array([cat_fit_by_pos[p] for p in applicants])
        applicant_skill_fit = compute_skill_fit(applicants, req_skill_idx, proficiency_matrix, neighbor_indices, gen_config.skill_relatedness_lambda)

        client_taste = taste_lookup.loc[row.client_id, taste_cols].to_numpy(dtype=float)
        raw_components = _raw_components(
            applicants, applicant_cat_fit, applicant_skill_fit, candidate_index.workers_df,
            workers_private, row.price_tier, client_taste, taste_cols,
        )

        standardized = {key: _standardize(raw_components[key], *moments[key]) for key in COMPONENT_KEYS}
        signal = sum(weights[_WEIGHT_KEY_BY_COMPONENT[key]] * standardized[key] for key in COMPONENT_KEYS)
        noise = rng_noise.normal(0, noise_std, size=len(applicants))
        utility = signal + noise

        # Shortlist: Plackett-Luce sample of shortlist_size without replacement.
        n_short = min(gen_config.shortlist_size, len(applicants))
        short_positions, short_utility = _plackett_luce_sample(
            rng_shortlist, applicants, utility, n_short, gen_config.shortlist_temperature
        )

        hire_prob = 1 / (1 + np.exp(-alpha * (short_utility.max() - theta)))
        hired_pos = None
        if rng_hire.random() < hire_prob:
            hire_probs = np.exp(short_utility / gen_config.hire_temperature)
            hire_probs = hire_probs / hire_probs.sum()
            hired_idx = rng_hire.choice(len(short_positions), p=hire_probs)
            hired_pos = short_positions[hired_idx]
            tracker.record_hire(int(hired_pos), day)

        for rank, (pos, u) in enumerate(zip(short_positions, short_utility), start=1):
            worker_id = worker_ids[pos]
            is_hired = hired_pos is not None and pos == hired_pos
            interaction_rows.append({
                "link": row.link, "worker_id": worker_id, "client_id": row.client_id,
                "simulated_day": day, "event": "HIRED" if is_hired else "SHORTLISTED",
                "rank_in_shortlist": rank,
            })
            pos_mask = applicants == pos
            truth_rows.append({
                "link": row.link, "worker_id": worker_id,
                "S_raw": float(raw_components["S"][pos_mask][0]), "T_raw": float(raw_components["T"][pos_mask][0]),
                "Q_raw": float(raw_components["Q"][pos_mask][0]), "B_raw": float(raw_components["B"][pos_mask][0]),
                "C_raw": float(raw_components["C"][pos_mask][0]),
                "S_std": float(standardized["S"][pos_mask][0]), "T_std": float(standardized["T"][pos_mask][0]),
                "Q_std": float(standardized["Q"][pos_mask][0]), "B_std": float(standardized["B"][pos_mask][0]),
                "C_std": float(standardized["C"][pos_mask][0]),
                "noise": float(noise[pos_mask][0]), "utility": float(u), "is_hired": bool(is_hired),
            })

    interactions_df = pd.DataFrame(interaction_rows)
    truth_df = pd.DataFrame(truth_rows)
    return interactions_df, truth_df


def _plackett_luce_sample(
    rng: np.random.Generator, positions: np.ndarray, utility: np.ndarray, n: int, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n` positions without replacement, probability
    proportional to exp(utility / temperature), Plackett-Luce style
    (sequential proportional-to-remaining-weight draws)."""
    remaining_pos = positions.copy()
    remaining_u = utility.copy()
    chosen_pos = []
    chosen_u = []
    for _ in range(n):
        weights = np.exp(remaining_u / temperature)
        weights = weights / weights.sum()
        idx = rng.choice(len(remaining_pos), p=weights)
        chosen_pos.append(remaining_pos[idx])
        chosen_u.append(remaining_u[idx])
        remaining_pos = np.delete(remaining_pos, idx)
        remaining_u = np.delete(remaining_u, idx)
    return np.array(chosen_pos), np.array(chosen_u)


# ---------------------------------------------------------------------------
# Ratings (docs Section 10)
# ---------------------------------------------------------------------------

def generate_ratings(truth_df: pd.DataFrame, interactions_df: pd.DataFrame, config: DataConfig, gen_config: GeneratorConfig) -> pd.DataFrame:
    """Ratings are conditional on HIRED interactions, stochastic, and
    derived from standardized quality/skill/budget truth plus noise --
    never a deterministic copy of latent quality."""
    hired = truth_df.loc[truth_df["is_hired"]].copy()
    rng_complete = stage_rng(config.seed, "rating")
    # A separate stream from rng_complete so completion and rating-noise
    # draws don't consume from (and thus perturb) the same sequence.
    rng_noise = np.random.default_rng(np.random.SeedSequence([config.seed, 0x8A71E5]))
    rng_review = stage_rng(config.seed, "review")

    completed_mask = rng_complete.random(len(hired)) < gen_config.rating_completion_rate
    completed = hired.loc[completed_mask].copy()
    if completed.empty:
        return pd.DataFrame(columns=["link", "worker_id", "rating", "has_review"])

    w = gen_config.rating_weights
    r_star = (
        w["quality"] * completed["Q_std"] + w["skill"] * completed["S_std"] + w["budget"] * completed["B_std"]
        + rng_noise.normal(0, gen_config.rating_noise_std, size=len(completed))
    )
    # Map to 1-5 via quantile thresholds chosen for a J-shaped distribution
    # skewed toward high ratings (approx 65% five-star, 20% four-star, 15%
    # three-or-below) [ASSUMPTION], applied to this run's realised r_star.
    q35, q15 = np.quantile(r_star, [0.35, 0.15])
    rating = np.select(
        [r_star >= q35, r_star >= q15],
        [5, 4],
        default=3,
    )
    # Push the bottom of the "3 or below" band down to 1-2 for variety.
    low_mask = rating == 3
    low_rank = pd.Series(r_star).rank(method="first")
    rating = np.where(low_mask & (low_rank <= low_mask.sum() * 0.3), 2, rating)
    rating = np.where(low_mask & (low_rank <= low_mask.sum() * 0.1), 1, rating)

    has_review = rng_review.random(len(completed)) < gen_config.review_rate

    return pd.DataFrame({
        "link": completed["link"].values,
        "worker_id": completed["worker_id"].values,
        "rating": rating.astype(int),
        "has_review": has_review,
    })


# ---------------------------------------------------------------------------
# Sparse-history threshold (docs Section 12)
# ---------------------------------------------------------------------------

def compute_sparse_threshold(
    interactions_df: pd.DataFrame, split_df: pd.DataFrame, workers_df: pd.DataFrame
) -> tuple[int, pd.Series]:
    """N = 33rd percentile of training hire counts among established
    (non-cold-start) workers, computed once, before any recommender runs.
    Returns (N, per-worker training hire counts for all established
    workers, including zero)."""
    train_links = set(split_df.loc[split_df["split"] == "train", "link"])
    hires = interactions_df.loc[(interactions_df["event"] == "HIRED") & interactions_df["link"].isin(train_links)]
    hire_counts = hires.groupby("worker_id").size()

    established_ids = workers_df.loc[~workers_df["is_cold_start"], "worker_id"]
    counts = hire_counts.reindex(established_ids, fill_value=0)
    n_threshold = int(np.percentile(counts, 33))
    return n_threshold, counts
