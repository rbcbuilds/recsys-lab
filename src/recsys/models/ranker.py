"""Learning-to-rank re-ranker with LightGBM  [SCAFFOLD — Phase 2].

Retrieval (popularity / item-CF / ALS / two-tower) gives you a few hundred
candidates optimized for *recall*. A ranker then re-orders them for *precision*
using rich features. This two-stage pattern (retrieve → rank) is the workhorse
of most production recommenders.

Feature ideas (per user-item candidate pair):
  * retrieval scores from the models above (great features!)
  * user aggregates: activity count, avg rating, recency
  * item aggregates: popularity, avg rating, age
  * content similarity (from text/image embeddings)
  * context: hour, day-of-week, location distance (Yelp has geo!)

Use LightGBM's ``lambdarank`` objective with ``group`` = per-user candidate
counts, and evaluate with NDCG@K. Train label = whether the candidate was a
real future positive (from the temporal split).

References: Burges 2010 (LambdaMART); the RecSys "retrieve then rank" pattern.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Recommender


class LightGBMRanker(Recommender):
    name = "lgbm_ranker"

    def __init__(self, num_leaves: int = 63, n_estimators: int = 200):
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators

    def fit(self, train: pd.DataFrame) -> "LightGBMRanker":
        raise NotImplementedError(
            "LTR re-ranker is a Phase-2 exercise. Generate candidates from a "
            "retrieval model, build user/item/context features, and train "
            "LightGBM with objective='lambdarank'. See this module's docstring."
        )

    def recommend(
        self, users: List[str], k: int = 10, exclude_seen: bool = True
    ) -> Dict[str, List[str]]:
        raise NotImplementedError
