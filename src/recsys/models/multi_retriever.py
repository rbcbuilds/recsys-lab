"""MultiRetriever — combine several candidate generators into one retriever.

Production retrieval is rarely a single model: separate generators (collaborative,
sequential, content, social, popularity) each capture a different notion of
relevance, and their candidates are unioned before a ranker sorts them. This
wraps any list of ``Recommender`` retrievers and fuses their ranked lists.

Fusion = **Reciprocal Rank Fusion (RRF)**::

    score(item) = Σ_retriever  weight_r / (rrf_k + rank_r(item))

RRF needs no score calibration across retrievers (it uses ranks only), which is
exactly why it is a common, robust default for merging heterogeneous sources.

Usable as the ``retriever=`` of ``TwoStageRecommender``: the ranker then re-orders
the fused candidate pool.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from .base import Recommender


class MultiRetriever(Recommender):
    name = "multi_retriever"

    def __init__(
        self,
        retrievers: Sequence[Recommender],
        weights: Optional[Sequence[float]] = None,
        rrf_k: int = 60,
        per_retriever: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        retrievers:
            Already-constructed retriever instances (content retrievers must be
            built with ``items=`` so their own ``fit`` has what it needs).
        weights:
            Per-retriever weight in the RRF sum (default: all 1.0).
        rrf_k:
            RRF smoothing constant. Larger = flatter contribution across ranks.
        per_retriever:
            How many candidates to pull from each source before fusing. Defaults
            to the requested ``k`` at recommend time.
        """
        if not retrievers:
            raise ValueError("MultiRetriever needs at least one retriever.")
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("weights must align with retrievers.")
        self.rrf_k = rrf_k
        self.per_retriever = per_retriever

    def fit(self, train: pd.DataFrame) -> "MultiRetriever":
        for r in self.retrievers:
            r.fit(train)
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        fetch = self.per_retriever or max(k, 50)
        # Collect each retriever's ranked lists once (batched over users).
        per_source: List[Dict[str, List[str]]] = [
            r.recommend(users, k=fetch, exclude_seen=exclude_seen) for r in self.retrievers
        ]

        out: Dict[str, List[str]] = {}
        for user_id in users:
            fused: Dict[str, float] = {}
            for w, recs in zip(self.weights, per_source):
                for rank, item in enumerate(recs.get(user_id, [])):
                    fused[item] = fused.get(item, 0.0) + w / (self.rrf_k + rank + 1)
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
            out[user_id] = [item for item, _ in ranked[:k]]
        return out
