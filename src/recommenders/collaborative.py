"""Collaborative Filtering recommender (Day 5 brief).

Learns purely from the collaborative structure of TRAINING-period
client-worker interactions -- never content (declared skills, category,
price, country -- see src/recommenders/content_based.py, which this
module does not import) and never generator-private data.

Two interpretable neighborhood methods, both operating on the same
sparse client x worker training-interaction matrix:

- **User-based**: client-client cosine similarity on interaction
  vectors; score a candidate worker as the similarity-weighted sum of
  how much the target client's nearest positively-similar *clients*
  interacted with that worker.
- **Item-based** (worker-based): worker-worker cosine similarity on
  "which clients interacted with this worker" vectors; score a
  candidate worker as the similarity-weighted sum, over the *target
  client's own* training history, restricted to that candidate's
  nearest positively-similar workers.

When a client has no training history at all, or a specific candidate
has no collaborative evidence (score exactly 0), ranking falls back to
Day 4's training-only Popularity signal (composed, not duplicated) --
see `CollaborativeFiltering.rank_with_diagnostics` for exactly how the
two are combined and how fallback usage is tracked.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from src.recommenders.base import BaseRecommender
from src.recommenders.popularity import PopularityBaseline


@dataclass
class TrainingMatrix:
    matrix: csr_matrix  # shape (n_clients, n_workers), TRAINING interactions only
    client_index: dict[str, int]
    worker_index: dict[str, int]
    clients: list[str] = field(default_factory=list)   # index -> client_id
    workers: list[str] = field(default_factory=list)    # index -> worker_id


def build_training_matrix(
    interactions: pd.DataFrame, requests: pd.DataFrame, event_weights: dict[str, float], aggregation: str
) -> TrainingMatrix:
    """Sparse client x worker matrix built exclusively from
    `split == "train"` interactions (Day 5 brief Section 2). Repeated
    client-worker pairs (the same client interacting with the same
    worker across multiple distinct training requests) are aggregated
    deterministically: "sum" totals the weight across all such
    interactions, "max" keeps only the single strongest observed
    signal."""
    if aggregation not in ("sum", "max"):
        raise ValueError(f"unsupported aggregation: {aggregation}")

    train_links = set(requests.loc[requests["split"] == "train", "link"])
    train = interactions.loc[interactions["link"].isin(train_links) & interactions["event"].isin(event_weights)].copy()
    train["weight"] = train["event"].map(event_weights)

    grouped = train.groupby(["client_id", "worker_id"])["weight"]
    aggregated = grouped.sum() if aggregation == "sum" else grouped.max()

    clients = sorted(aggregated.index.get_level_values(0).unique())
    workers = sorted(aggregated.index.get_level_values(1).unique())
    client_index = {c: i for i, c in enumerate(clients)}
    worker_index = {w: i for i, w in enumerate(workers)}

    rows = [client_index[c] for c, _ in aggregated.index]
    cols = [worker_index[w] for _, w in aggregated.index]
    matrix = csr_matrix((aggregated.to_numpy(dtype=float), (rows, cols)), shape=(len(clients), len(workers)))
    return TrainingMatrix(matrix=matrix, client_index=client_index, worker_index=worker_index, clients=clients, workers=workers)


def cosine_similarity_matrix(matrix: csr_matrix) -> np.ndarray:
    """Dense row-wise cosine similarity for `matrix` (rows are the
    entities being compared). A zero row (should not occur here, since
    only entities with >=1 training interaction are included in the
    matrix) would map to a zero similarity row rather than raising."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = diags(1.0 / safe_norms) @ matrix
    similarity = (normalized @ normalized.T).toarray()
    similarity[norms == 0, :] = 0.0
    similarity[:, norms == 0] = 0.0
    return similarity


def build_neighbor_lists(similarity: np.ndarray, min_similarity: float) -> list[np.ndarray]:
    """For each row i, indices of other rows sorted by descending
    similarity, self excluded, restricted to similarity > min_similarity
    ("only positive useful collaborative evidence" -- Day 5 brief
    Section 7). Precomputed once per similarity matrix so evaluating
    multiple neighborhood sizes K does not require re-sorting."""
    n = similarity.shape[0]
    neighbor_lists = []
    for i in range(n):
        row = similarity[i].copy()
        row[i] = -np.inf  # self-exclusion
        order = np.argsort(-row, kind="stable")
        order = order[row[order] > min_similarity]
        neighbor_lists.append(order)
    return neighbor_lists


class CollaborativeFiltering(BaseRecommender):
    name = "cf"

    def __init__(
        self,
        training_matrix: TrainingMatrix,
        method: str,
        k_neighbors: int,
        fallback: PopularityBaseline,
        link_to_client: dict[str, str],
        min_similarity: float = 0.0,
    ):
        if method not in ("user", "item"):
            raise ValueError(f"method must be 'user' or 'item', got {method!r}")
        self.matrix = training_matrix
        self.method = method
        self.k_neighbors = k_neighbors
        self.fallback = fallback
        self.link_to_client = link_to_client
        self.min_similarity = min_similarity

        if method == "user":
            self.similarity = cosine_similarity_matrix(training_matrix.matrix)
        else:
            self.similarity = cosine_similarity_matrix(training_matrix.matrix.T.tocsr())
        self.neighbor_lists = build_neighbor_lists(self.similarity, min_similarity)

    def _cf_scores(self, link: str, candidates: list[str]) -> dict[str, float]:
        """Raw collaborative score per candidate worker. A worker absent
        from the returned dict has no collaborative evidence (treated as
        score 0 by the caller)."""
        client_id = self.link_to_client.get(link)
        if client_id is None or client_id not in self.matrix.client_index:
            return {}  # no training history for this client at all

        if self.method == "user":
            return self._score_user_based(client_id, candidates)
        return self._score_item_based(client_id, candidates)

    def _score_user_based(self, client_id: str, candidates: list[str]) -> dict[str, float]:
        client_idx = self.matrix.client_index[client_id]
        neighbor_clients = self.neighbor_lists[client_idx][: self.k_neighbors]
        if len(neighbor_clients) == 0:
            return {}

        sims = self.similarity[client_idx, neighbor_clients]
        candidate_cols = [self.matrix.worker_index[w] for w in candidates if w in self.matrix.worker_index]
        if not candidate_cols:
            return {}

        sub = self.matrix.matrix[neighbor_clients][:, candidate_cols]  # (n_neighbors, n_candidates)
        scores = np.asarray(sims @ sub).ravel()

        candidate_with_col = [w for w in candidates if w in self.matrix.worker_index]
        return {w: float(s) for w, s in zip(candidate_with_col, scores)}

    def _score_item_based(self, client_id: str, candidates: list[str]) -> dict[str, float]:
        client_idx = self.matrix.client_index[client_id]
        client_row = self.matrix.matrix[client_idx]
        history_cols = client_row.indices
        if len(history_cols) == 0:
            return {}
        history_weight = dict(zip(history_cols.tolist(), client_row.data.tolist()))
        history_set = set(history_cols.tolist())

        result = {}
        for worker_id in candidates:
            worker_idx = self.matrix.worker_index.get(worker_id)
            if worker_idx is None:
                continue
            neighbor_workers = self.neighbor_lists[worker_idx][: self.k_neighbors]
            relevant = [n for n in neighbor_workers.tolist() if n in history_set]
            if not relevant:
                result[worker_id] = 0.0
                continue
            sims = self.similarity[worker_idx, relevant]
            weights = np.array([history_weight[n] for n in relevant])
            result[worker_id] = float(np.dot(sims, weights))
        return result

    def rank_with_diagnostics(self, link: str, candidates: list[str], k: int) -> tuple[list[str], dict]:
        cf_scores = self._cf_scores(link, candidates)
        client_id = self.link_to_client.get(link)
        client_has_history = client_id is not None and client_id in self.matrix.client_index

        fallback_scores = {w: self.fallback.hire_counts.get(w, 0) for w in candidates}
        ranked = sorted(
            candidates,
            key=lambda w: (-cf_scores.get(w, 0.0), -fallback_scores[w], w),
        )[:k]

        n_via_fallback = sum(1 for w in ranked if cf_scores.get(w, 0.0) <= 0.0)
        diagnostics = {
            "client_has_history": client_has_history,
            "request_fully_fallback": not client_has_history or all(cf_scores.get(w, 0.0) <= 0.0 for w in candidates),
            "n_recommendations": len(ranked),
            "n_recommendations_via_fallback": n_via_fallback,
        }
        return ranked, diagnostics

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        ranked, _ = self.rank_with_diagnostics(link, candidates, k)
        return ranked
