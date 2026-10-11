"""Request-level paired bootstrap significance (docs/experimental_design.md
Section 15: "Paired bootstrap 95% confidence intervals (1,000 resamples
over test requests) for differences between the hybrid and each
baseline").

For per-request metric vectors a (method A) and b (method B) over the
SAME requests, d = a - b. Each bootstrap replicate resamples requests with
replacement and records mean(d). Reported:

- mean_difference: mean(d) on the observed requests;
- ci_low / ci_high: percentile interval of the replicate means;
- p_value: two-sided bootstrap p-value under the null of zero mean
  difference, by the shift method: replicate means are centred at 0
  (d* - mean(d)) and p = (1 + #{|centred| >= |mean(d)|}) / (B + 1). The
  +1 terms keep p > 0 and make identical methods give p = 1.

Resampling indices come from `numpy.random.SeedSequence(seed)`, so results
are deterministic; every comparison with the same seed and request count
uses the same resamples (common random numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_indices(n: int, n_resamples: int, seed) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return rng.integers(0, n, size=(n_resamples, n))


def paired_bootstrap(a, b, n_resamples: int, seed, confidence_level: float = 0.95) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired vectors must be 1-D and the same length, got {a.shape} and {b.shape}")
    n = len(a)
    if n == 0:
        raise ValueError("cannot bootstrap zero paired observations")
    d = a - b
    observed = float(d.mean())
    replicate_means = d[bootstrap_indices(n, n_resamples, seed)].mean(axis=1)
    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(replicate_means, [alpha / 2, 1 - alpha / 2])
    extreme = int(np.sum(np.abs(replicate_means - observed) >= abs(observed)))
    return {
        "n_pairs": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": (1 + extreme) / (n_resamples + 1),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def align_per_request(per_request_a: pd.DataFrame, per_request_b: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pair two evaluator per-request frames by `link`. Both must cover
    exactly the same requests (same protocol, same exclusions)."""
    links_a, links_b = set(per_request_a["link"]), set(per_request_b["link"])
    if links_a != links_b or len(links_a) != len(per_request_a) or len(links_b) != len(per_request_b):
        raise ValueError("per-request frames must cover exactly the same, unique requests")
    links = sorted(links_a)
    a = per_request_a.set_index("link").loc[links, metric].to_numpy(dtype=float)
    b = per_request_b.set_index("link").loc[links, metric].to_numpy(dtype=float)
    return a, b, links
