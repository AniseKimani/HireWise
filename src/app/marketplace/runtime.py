"""Runtime recommendations for client-created requests (Day 12).

A SEPARATE path from the frozen experimental evaluation. It applies the
frozen Day 8 hybrid -- same components, same selected weights
(CBF 0.40 / CF 0.10 / KG 0.50), same per-pool min-max normalisation --
to a new request described only by observable features. Nothing is
retrained, and no frozen object is modified:

- Feature mapping. The request is put into the experimental request
  format: category, controlled-vocabulary skills, location requirement,
  and `price_tier` computed with the exact Day 2 definition (pandas
  average-rank percentile of the pay amount within (category, pay_type)
  over the frozen in-scope reference population -- validated to
  reproduce the stored tiers exactly by leave-one-out).
- Timeline. The request is placed at the END of the simulated timeline
  (the latest experimental request day): every catalogue worker has
  joined, and the frozen KG temporal guard holds -- only the
  training-period history in the frozen snapshot is visible.
- CBF: a NEW frozen-weight ContentBasedRecommender over the frozen worker
  profiles and this one request.
- KG: a NEW KnowledgeGraphRecommender over the FROZEN training-only
  KGIndex with the frozen selected configuration. Skill, related-skill,
  category and reputation paths need only observable features. History
  paths exist only if the client account is linked to an imported client
  with real training hires; a newly registered client has none, so none
  are fabricated.
- CF: a shallow, non-mutating view of the frozen CF model whose only
  difference is the link -> client map for this request. Without an
  imported client history the CF score is constant (0) and, under the
  hybrid's existing constant-component rule, contributes exactly 0.
- Candidates: the experimental eligibility function over the imported
  catalogue. Newly registered worker accounts are not in the frozen
  models and are therefore not recommended (documented limitation).
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.serializers import serialize_recommendation, serialize_worker
from src.data.clean import _pay_amount
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn
from src.recommenders.knowledge_graph import KGComponentCache, KnowledgeGraphRecommender

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPARED_JOBS = REPO_ROOT / "data" / "processed" / "prepared_jobs.csv"
RUNTIME_LINK_PREFIX = "runtime:"
RUNTIME_CLIENT_PREFIX = "runtime-user:"


class PriceTierReference:
    """Day 2 price-tier definition for a job outside the reference population."""

    def __init__(self, groups: dict[tuple[str, str], np.ndarray]):
        self._groups = groups

    @classmethod
    def from_prepared_jobs(cls, path: Path = PREPARED_JOBS) -> "PriceTierReference":
        df = pd.read_csv(path, usecols=["category", "pay_type", "pay_amount", "is_in_scope_category", "is_duplicate", "has_no_skills"])
        ref = df.loc[df["is_in_scope_category"] & ~df["is_duplicate"] & ~df["has_no_skills"] & df["pay_amount"].notna()]
        return cls({key: np.sort(g["pay_amount"].to_numpy(dtype=float)) for key, g in ref.groupby(["category", "pay_type"])})

    def tier(self, category: str, pay_type: str, budget, hourly_low, hourly_high) -> float | None:
        amount = _pay_amount({"pay_type": pay_type, "budget": budget, "hourly_low": hourly_low, "hourly_high": hourly_high})
        values = self._groups.get((category, pay_type))
        if amount is None or values is None:
            return None  # unspecified pay or no reference group: CBF's neutral price score applies
        less = int(np.searchsorted(values, amount, side="left"))
        equal = int(np.searchsorted(values, amount, side="right")) - less
        # pandas rank(pct=True, method="average") of `amount` were it added to the group
        return (less + (equal + 2) / 2) / (len(values) + 1)


@dataclass(frozen=True)
class RuntimeRequest:
    request_id: int
    client_user_id: int
    category: str
    skills: tuple[str, ...]
    price_tier: float | None
    location_requirement: str | None
    imported_client_id: str | None  # set only for demo accounts linked to an imported client

    @property
    def link(self) -> str:
        return f"{RUNTIME_LINK_PREFIX}{self.request_id}"


class RuntimeRecommender:
    def __init__(self, service, price_tiers: PriceTierReference):
        self.service = service
        self.price_tiers = price_tiers
        self.weights = dict(service.model_version.weights)
        # End of the simulated timeline (latest experimental request day).
        self.runtime_day = float(service.data.requests["simulated_day"].max())
        self._workers = service.data.workers
        self._lock = threading.Lock()

    @classmethod
    def build(cls, service) -> "RuntimeRecommender":
        return cls(service, PriceTierReference.from_prepared_jobs())

    def candidates(self, request: RuntimeRequest) -> list[str]:
        return self.service.eligibility.eligible_worker_ids(request.category, self.runtime_day, request.location_requirement)

    def recommend(self, request: RuntimeRequest, k: int, kg_evidence_paths: int = 3) -> dict:
        models = self.service.models
        frozen_cbf, frozen_cf, frozen_kg = models["cbf"], models["cf"], models["kg"]
        link = request.link
        kg_client = request.imported_client_id or f"{RUNTIME_CLIENT_PREFIX}{request.client_user_id}"
        row = pd.DataFrame([{
            "link": link, "client_id": kg_client, "category": request.category, "simulated_day": self.runtime_day,
            "skills_vocab": "|".join(request.skills),
            "price_tier": float("nan") if request.price_tier is None else request.price_tier,
            "location_requirement": request.location_requirement,
        }])
        cbf = ContentBasedRecommender(self._workers, row, frozen_cbf.weights, frozen_cbf.price_neutral_score)
        kg = KnowledgeGraphRecommender(frozen_kg.index, row, frozen_kg.damping, frozen_kg.weights, frozen_kg.role_weights,
                                       frozen_kg.reputation_prior_strength, frozen_kg.max_paths_per_component,
                                       KGComponentCache((frozen_kg.damping,)))
        cf_view = copy.copy(frozen_cf)  # shares the frozen matrices read-only
        cf_view.link_to_client = {link: request.imported_client_id} if request.imported_client_id else {}
        hybrid = HybridRecommender({"cbf": cbf_score_fn(cbf), "cf": cf_score_fn(cf_view), "kg": kg_score_fn(kg)}, self.weights)
        candidates = self.candidates(request)
        with self._lock:
            explanations = hybrid.explain(link, candidates, k, kg_recommender=kg, kg_paths=kg_evidence_paths)
        profiles = self.service.repository.get_workers([e.worker_id for e in explanations])
        return {
            "request_id": request.request_id,
            "k": k,
            "n_candidates": len(candidates),
            "n_returned": len(explanations),
            "model": self.service.model_version.to_dict(),
            "history": {
                "client_linked_to_imported_profile": request.imported_client_id is not None,
                "collaborative_history_available": request.imported_client_id in frozen_cf.matrix.client_index,
                "kg_hiring_history_available": bool(frozen_kg.index.hired.get(kg_client)),
            },
            "recommendations": [
                serialize_recommendation(e, serialize_worker(profiles[e.worker_id], profiles[e.worker_id]["review_count"],
                                                             profiles[e.worker_id]["review_sum"]), kg_evidence_paths)
                for e in explanations
            ],
        }
