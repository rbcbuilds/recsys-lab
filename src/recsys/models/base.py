"""Base recommender interface + shared index/utility helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import settings


class Recommender(ABC):
    """Common interface for every model in the lab.

    Subclasses implement :meth:`fit` and :meth:`recommend`. Keeping one
    interface means the same evaluation loop works for popularity, ALS, a
    two-tower net, or an LLM re-ranker.
    """

    name: str = "recommender"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Train on long-format interactions [user_id, item_id, rating, timestamp]."""

    @abstractmethod
    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        """Return ``{user_id: [item_id, ...]}`` top-K ranked recommendations."""

    def recommend_all(self, k: int = 10, exclude_seen: bool = True) -> Dict[str, List[str]]:
        return self.recommend(list(self.user_ids), k=k, exclude_seen=exclude_seen)


class IndexedRecommender(Recommender):
    """Base class that builds integer id<->index maps and a seen-items lookup.

    Most collaborative models need a contiguous integer index over users/items
    and the set of items each user already interacted with (to exclude them
    from recommendations). This centralizes that bookkeeping.
    """

    def _build_index(self, train: pd.DataFrame) -> None:
        u, i = settings.user_col, settings.item_col
        self.user_ids = train[u].unique().tolist()
        self.item_ids = train[i].unique().tolist()
        self.user_to_idx = {uid: n for n, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: n for n, iid in enumerate(self.item_ids)}
        self.idx_to_item = np.array(self.item_ids)

        self.seen: Dict[str, set] = (
            train.groupby(u)[i].agg(set).to_dict()
        )

    def _top_k_from_scores(
        self, user_id: str, scores: np.ndarray, k: int, exclude_seen: bool
    ) -> List[str]:
        """Turn a (n_items,) score vector into top-K item ids."""
        if exclude_seen:
            for it in self.seen.get(user_id, ()):  # mask seen items
                idx = self.item_to_idx.get(it)
                if idx is not None:
                    scores[idx] = -np.inf
        # argpartition for speed, then sort the top slice
        n = min(k, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [self.idx_to_item[j] for j in top if np.isfinite(scores[j])]
