"""Popularity / trending baseline.

The simplest possible recommender and a surprisingly strong one. Also your
cold-start fallback: for a brand-new user with no history, "show what's popular"
is often the best you can do. Never skip this — it sets the floor every fancier
model must beat.

Two flavors:
* ``recency_weighted=False``: rank items by raw interaction count.
* ``recency_weighted=True``: weight recent interactions more (trending), using
  an exponential decay on age in days.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import settings
from .base import IndexedRecommender


class PopularityRecommender(IndexedRecommender):
    name = "popularity"

    def __init__(self, recency_weighted: bool = False, half_life_days: float = 90.0):
        self.recency_weighted = recency_weighted
        self.half_life_days = half_life_days
        self._ranked_items: List[str] = []

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        self._build_index(train)
        i, t = settings.item_col, settings.time_col

        if self.recency_weighted:
            most_recent = train[t].max()
            age_days = (most_recent - train[t]).dt.total_seconds() / 86400.0
            decay = np.power(0.5, age_days / self.half_life_days)
            scores = train.assign(_w=decay.values).groupby(i)["_w"].sum()
        else:
            scores = train.groupby(i).size()

        self._ranked_items = scores.sort_values(ascending=False).index.tolist()
        self._item_score = scores.to_dict()
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for user_id in users:
            if exclude_seen:
                seen = self.seen.get(user_id, set())
                recs = [it for it in self._ranked_items if it not in seen][:k]
            else:
                recs = self._ranked_items[:k]
            out[user_id] = recs
        return out
