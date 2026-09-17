"""Synthetic client generation and request assignment
(docs/experimental_design.md Section 7).

Clients are synthetic because the raw Upwork dataset has no client
identifier. To keep them grounded, the 12,000 real Day 2 sampled requests
are distributed among 2,000 synthetic clients rather than inventing new
demand: client counts per country, primary category choice, and activity
level are all derived from the real request distribution, and every
request keeps its original `link` so it stays traceable to the Day 2
processed record.

Latent taste `u_c ~ N(0, I_8)` is generator-private (returned separately
in `clients_private_df`) and must never be joined into `clients_df`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import DataConfig
from src.data.generator_config import GeneratorConfig
from src.data.synth_rng import stage_rng
from src.data.synth_util import apportion_with_floor, stable_id

UNSPECIFIED_COUNTRY = "Unspecified"


def _client_counts_per_country(request_country_counts: pd.Series, n_clients: int) -> pd.Series:
    counts = apportion_with_floor(request_country_counts, total=n_clients, floor=1)

    # Safety net: a client must own >= 1 request, so a country can never
    # get more clients than it has requests. Largest-remainder apportion
    # with floor=1 will not normally violate this (a country's weight is
    # too small to win extra remainder seats), but we guard it explicitly
    # and park any freed seats on the largest country [IMPL].
    excess = 0
    for country in counts.index:
        cap = int(request_country_counts[country])
        if counts[country] > cap:
            excess += counts[country] - cap
            counts[country] = cap
    if excess > 0:
        biggest = request_country_counts.idxmax()
        counts[biggest] += excess
    return counts


def generate_clients(
    sampled_requests: pd.DataFrame, config: DataConfig, gen_config: GeneratorConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (clients_df, clients_private_df, assignment_df).

    clients_df: client_id, country, primary_category, activity_start_day,
        activity_quota (observable; quota is a generation target, realised
        counts may differ slightly after the leftover-reconciliation pass).
    clients_private_df: client_id, taste_0..taste_{dim-1} (generator-private).
    assignment_df: link, client_id -- which client owns which of the
        12,000 real sampled requests.
    """
    requests = sampled_requests[["link", "category", "country_clean"]].copy()
    requests["country_clean"] = requests["country_clean"].fillna(UNSPECIFIED_COUNTRY)

    country_counts = requests["country_clean"].value_counts()
    client_counts = _client_counts_per_country(country_counts, config.synthetic_clients)

    country_rng = stage_rng(config.seed, "client_country_allocation")
    category_rng = stage_rng(config.seed, "client_category")
    activity_rng = stage_rng(config.seed, "client_activity")
    assignment_rng = stage_rng(config.seed, "client_assignment")
    timeline_start_rng = stage_rng(config.seed, "timeline")

    client_rows = []
    client_id_counter = 0
    # Deterministic country order.
    for country in sorted(client_counts.index):
        n = int(client_counts[country])
        country_requests = requests.loc[requests["country_clean"] == country]
        category_dist = country_requests["category"].value_counts(normalize=True)
        categories = category_dist.index.to_numpy()
        probs = category_dist.to_numpy()

        primary_categories = category_rng.choice(categories, size=n, p=probs)
        for i in range(n):
            client_rows.append({
                "client_id": stable_id("C", client_id_counter),
                "country": country,
                "primary_category": primary_categories[i],
            })
            client_id_counter += 1

    clients_df = pd.DataFrame(client_rows)

    # Heavy-tailed activity weight per client (lognormal, mean calibrated
    # to config.client_activity_mean), clipped to [min, max]. These are
    # only *weights* used to split each country's requests unevenly
    # across its clients -- see the per-country apportionment below,
    # which is what actually determines each client's request count.
    sigma = 0.8
    mu = np.log(gen_config.client_activity_mean) - sigma**2 / 2
    raw_weights = activity_rng.lognormal(mean=mu, sigma=sigma, size=len(clients_df))
    raw_weights = np.clip(raw_weights, gen_config.client_activity_min, gen_config.client_activity_max)
    clients_df["_activity_weight"] = raw_weights

    quotas = pd.Series(0, index=clients_df["client_id"], dtype=int)
    for country in sorted(client_counts.index):
        mask = clients_df["country"] == country
        weights = clients_df.loc[mask].set_index("client_id")["_activity_weight"]
        n_requests = int(country_counts[country])
        country_quota = apportion_with_floor(weights, total=n_requests, floor=1)
        quotas.loc[country_quota.index] = country_quota.values

    clients_df["activity_quota"] = clients_df["client_id"].map(quotas)
    clients_df = clients_df.drop(columns=["_activity_weight"])

    clients_df["activity_start_day"] = timeline_start_rng.uniform(
        0, gen_config.client_start_day_max, size=len(clients_df)
    )

    # --- Assignment: fill each client's quota, ~coherence share from its
    # primary category, the rest from any request in the same country.
    remaining_by_country: dict[str, list[str]] = {
        country: requests.loc[requests["country_clean"] == country, "link"].sort_values().tolist()
        for country in client_counts.index
    }
    category_by_link = requests.set_index("link")["category"]

    assignment: dict[str, str] = {}
    for row in clients_df.sort_values("client_id").itertuples():
        pool = remaining_by_country[row.country]
        if not pool:
            continue
        quota = int(row.activity_quota)
        same_target = round(quota * gen_config.client_category_coherence)

        same_cat = [link for link in pool if category_by_link[link] == row.primary_category]
        n_same = min(same_target, len(same_cat), quota)
        picked_same = list(assignment_rng.choice(same_cat, size=n_same, replace=False)) if n_same else []

        pool_after_same = [link for link in pool if link not in picked_same]
        remaining_quota = quota - len(picked_same)
        n_other = min(remaining_quota, len(pool_after_same))
        picked_other = (
            list(assignment_rng.choice(pool_after_same, size=n_other, replace=False)) if n_other else []
        )

        picked = picked_same + picked_other
        for link in picked:
            assignment[link] = row.client_id
        remaining_by_country[row.country] = [link for link in pool_after_same if link not in picked_other]

    # Reconciliation: any requests left unassigned (quota rounding
    # shortfall) go round-robin to that country's clients, ignoring quota.
    for country, leftover in remaining_by_country.items():
        if not leftover:
            continue
        country_clients = clients_df.loc[clients_df["country"] == country, "client_id"].sort_values().tolist()
        for i, link in enumerate(leftover):
            assignment[link] = country_clients[i % len(country_clients)]

    assignment_df = pd.DataFrame({"link": list(assignment.keys()), "client_id": list(assignment.values())})
    assignment_df = assignment_df.sort_values("link").reset_index(drop=True)

    taste_rng = stage_rng(config.seed, "client_taste")
    taste_dim = gen_config.client_taste_dim
    taste = taste_rng.standard_normal(size=(len(clients_df), taste_dim))
    clients_private_df = pd.DataFrame(taste, columns=[f"taste_{i}" for i in range(taste_dim)])
    clients_private_df.insert(0, "client_id", clients_df["client_id"].values)

    clients_df = clients_df.sort_values("client_id").reset_index(drop=True)
    return clients_df, clients_private_df, assignment_df


def finalize_client_observables(
    clients_df: pd.DataFrame, assignment_df: pd.DataFrame, requests_df: pd.DataFrame
) -> pd.DataFrame:
    """Add observable attributes derived from each client's assigned real
    requests: preferred pay type (mode) and median price tier."""
    merged = assignment_df.merge(requests_df[["link", "pay_type", "price_tier"]], on="link", how="left")
    agg = merged.groupby("client_id").agg(
        preferred_pay_type=("pay_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "unspecified"),
        median_price_tier=("price_tier", "median"),
        n_requests_assigned=("link", "count"),
    )
    return clients_df.merge(agg, on="client_id", how="left")
