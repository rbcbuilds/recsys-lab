"""Bayesian Personalized Ranking (BPR) — same MF family as ALS, different loss.

ALS minimizes a pointwise (confidence-weighted) reconstruction error. BPR
optimizes a *pairwise ranking* objective: for each user, an observed item should
score above an unobserved one::

    L = Σ_{(u,i,j)} -log σ(x̂_ui - x̂_uj)

where i is observed for u and j is not. Same latent factors, different objective —
the clean A/B for \"does the loss function matter?\".

Uses ``implicit.bpr.BayesianPersonalizedRanking`` so the comparison against
``ALSRecommender`` is library-matched (same sparse matrix, same recommend API).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import settings
from .base import IndexedRecommender


class BPRRecommender(IndexedRecommender):
    name = "bpr"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        learning_rate: float = 0.05,
        iterations: int = 100,
        use_ratings: bool = False,
    ):
        """
        Parameters
        ----------
        use_ratings:
            BPR is a ranking objective over binary preferences. Default False
            (binary interaction). If True, ratings are used as matrix values
            (implicit still treats the matrix as preference confidence).
        """
        self.factors = factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.use_ratings = use_ratings
        self._model = None

    def fit(self, train: pd.DataFrame) -> "BPRRecommender":
        try:
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BPRRecommender needs the 'implicit' package. "
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
        self.ui = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            random_state=settings.seed,
        )
        # BPR expects a binary-ish user-item matrix (no alpha confidence scaling).
        self._model.fit(self.ui.astype(np.float32), show_progress=False)
        return self

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        if self._model is None:
            raise RuntimeError("Call fit() before recommend().")
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
