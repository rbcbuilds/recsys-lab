"""Matrix factorization via Alternating Least Squares (implicit feedback).

Learns low-dimensional user and item vectors whose dot product approximates
preference. The go-to strong baseline for implicit feedback (clicks, views,
reviews). Uses the ``implicit`` library's ALS, which handles large sparse
matrices efficiently with confidence-weighted implicit feedback.

If ``implicit`` isn't installed, this raises a clear message pointing at
requirements.txt.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import settings
from .base import IndexedRecommender


class ALSRecommender(IndexedRecommender):
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha: float = 40.0,
        use_ratings: bool = True,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_ratings = use_ratings
        self._model = None

    def fit(self, train: pd.DataFrame) -> "ALSRecommender":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ALSRecommender needs the 'implicit' package. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

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
        # implicit expects a user-item confidence matrix; alpha scales confidence.
        self.ui = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items)
        )
        confidence = (self.ui * self.alpha).astype(np.float32)

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=settings.seed,
        )
        self._model.fit(confidence, show_progress=False)
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for user_id in users:
            uidx = self.user_to_idx.get(user_id)
            if uidx is None:
                out[user_id] = []
                continue
            ids, _scores = self._model.recommend(
                uidx,
                self.ui[uidx],
                N=k,
                filter_already_liked_items=exclude_seen,
            )
            out[user_id] = [self.idx_to_item[j] for j in ids]
        return out
