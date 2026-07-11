"""Item-based collaborative filtering.

Intuition: "items co-liked by the same users are similar; recommend items
similar to what you already liked." Interpretable, strong on sparse data, and
the classic second baseline after popularity.

Implementation:
* Build a sparse user x item matrix (binary or rating-weighted).
* Compute item-item cosine similarity (optionally top-N neighbors only).
* Score a user's candidate items as the sum of similarities to items they've
  interacted with.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import settings
from .base import IndexedRecommender


class ItemCFRecommender(IndexedRecommender):
    name = "item_cf"

    def __init__(self, top_n_neighbors: int = 50, use_ratings: bool = False):
        self.top_n_neighbors = top_n_neighbors
        self.use_ratings = use_ratings

    def fit(self, train: pd.DataFrame) -> "ItemCFRecommender":
        self._build_index(train)
        u, i, r = settings.user_col, settings.item_col, settings.rating_col

        rows = train[u].map(self.user_to_idx).values
        cols = train[i].map(self.item_to_idx).values
        vals = (
            train[r].astype(float).values
            if self.use_ratings
            else np.ones(len(train), dtype=float)
        )
        n_users, n_items = len(self.user_ids), len(self.item_ids)
        self.ui = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # Cosine similarity between item columns.
        item_user = self.ui.T.tocsr()
        norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1e-8
        self._norms = norms
        self._item_user = item_user
        return self

    def _similarities(self, item_idx: int) -> np.ndarray:
        # cosine(i, all) = (i . all) / (|i| |all|)
        dots = self._item_user @ self._item_user[item_idx].T
        dots = np.asarray(dots.todense()).ravel()
        sims = dots / (self._norms * self._norms[item_idx])
        sims[item_idx] = 0.0
        if self.top_n_neighbors and self.top_n_neighbors < len(sims):
            cutoff_idx = np.argpartition(-sims, self.top_n_neighbors)[
                self.top_n_neighbors :
            ]
            sims[cutoff_idx] = 0.0
        return sims

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        sim_cache: Dict[int, np.ndarray] = {}

        for user_id in users:
            uidx = self.user_to_idx.get(user_id)
            if uidx is None:
                out[user_id] = []
                continue
            liked = self.ui[uidx].indices
            if len(liked) == 0:
                out[user_id] = []
                continue
            scores = np.zeros(len(self.item_ids), dtype=float)
            for it in liked:
                if it not in sim_cache:
                    sim_cache[it] = self._similarities(it)
                scores += sim_cache[it]
            out[user_id] = self._top_k_from_scores(user_id, scores, k, exclude_seen)
        return out
